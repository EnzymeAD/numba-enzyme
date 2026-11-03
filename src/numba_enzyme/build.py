"""
Handles the pipeline from lowering the function to building the native object.

Wires lowering, driver synthesis, in-process LLVM IR extraction/linking and
Enzyme differentiation via `libNumbaEnzyme`, and `llvmlite.binding` native
object emission into a single call, caching the built object files on disk.

See Also
--------
numba_enzyme.runtime.load : Loads the `BuiltKernel` this module produces.
numba_enzyme.core.grad : Public API built on top of this module.

Examples
--------
>>> from numba_enzyme.build import build
>>> from numba_enzyme.types import Float64
>>> def f(x: Float64) -> Float64:
...     return x * x
>>> build(f).n_args  # doctest: +SKIP
1
"""

import ctypes
import hashlib
import inspect
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import llvmlite
import llvmlite.binding as llvm_binding
import numba as nb

from numba_enzyme.driver import synthesise
from numba_enzyme.lowering import ArgSpec, lower
from numba_enzyme.toolchain import get_toolchain

_CACHE_DIR_ENV_VAR = "NUMBA_ENZYME_CACHE_DIR"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "numba_enzyme"


@dataclass(frozen=True)
class BuiltKernel:
    """
    A compiled, cacheable native object file exposing grad/JVP entry points.

    Attributes
    ----------
    path : pathlib.Path
        Path to the built native object file (`kernel.o`).
    grad_symbol : str
        Name of the reverse-mode entry point exported by `path`.
    jvp_symbol : str
        Name of the forward-mode entry point exported by `path`.
    n_args : int
        Number of logical arguments the original function takes.
    arg_specs : tuple[ArgSpec, ...]
        ABI specification for each logical argument.
    from_cache : bool
        Whether the object file was loaded from the disk cache.

    See Also
    --------
    build : Builds and validates a `BuiltKernel` instance.

    Examples
    --------
    >>> from numba_enzyme.build import build
    >>> from numba_enzyme.types import Float64
    >>> def f(x: Float64) -> Float64:
    ...     return x * x
    >>> build(f).from_cache  # doctest: +SKIP
    False
    """

    path: Path
    grad_symbol: str
    jvp_symbol: str
    n_args: int
    arg_specs: tuple[ArgSpec, ...]
    from_cache: bool


def _cache_dir() -> Path:
    """
    Path in which the cached compiled kernels are stored.

    Defaults to `~/.cache/numba_enzyme`, overridable via the
    `NUMBA_ENZYME_CACHE_DIR` environment variable.

    Returns
    -------
    pathlib.Path
        The cache root directory.

    Examples
    --------
    >>> from numba_enzyme.build import _cache_dir
    >>> _cache_dir()  # doctest: +SKIP
    PosixPath('/home/user/.cache/numba_enzyme')
    """
    override = os.environ.get(_CACHE_DIR_ENV_VAR)
    return Path(override) if override else _DEFAULT_CACHE_DIR


def _toolchain_fingerprint() -> str:
    """
    Fingerprint `libNumbaEnzyme`, `llvmlite`, `numba`, and the host target.

    Used as part of the cache key. Rebuilding `libNumbaEnzyme` or upgrading
    `numba`/`llvmlite` invalidates cached entries.

    Returns
    -------
    str
        A string encoding `libNumbaEnzyme`'s path, mtime, and size together
        with the active `llvmlite`, `numba`, and target triple metadata.

    See Also
    --------
    numba_enzyme.toolchain.get_toolchain : Resolves `libNumbaEnzyme`.

    Examples
    --------
    >>> from numba_enzyme.build import _toolchain_fingerprint
    >>> _toolchain_fingerprint()  # doctest: +SKIP
    '.../libNumbaEnzyme.so:...|llvmlite=0.50.0|numba=0.68.1|...'
    """
    tc = get_toolchain()
    stat = tc.enzyme_lib.stat()
    return (
        f"{tc.enzyme_lib}:{stat.st_mtime_ns}:{stat.st_size}"
        f"|llvmlite={llvmlite.__version__}"
        f"|numba={nb.__version__}"
        f"|triple={llvm_binding.get_process_triple()}"
        f"|cpu={llvm_binding.get_host_cpu_name()}"
    )


def _cache_key(func: Callable) -> str:
    """
    Compute `build`'s cache key for a function.

    Parameters
    ----------
    func : callable
        The function to key on.

    Returns
    -------
    str
        A hex-encoded SHA-256 digest of `func`'s source text combined
        with the current `_toolchain_fingerprint`.

    Examples
    --------
    >>> from numba_enzyme.build import _cache_key
    >>> from numba_enzyme.types import Float64
    >>> def f(x: Float64) -> Float64:
    ...     return x * x
    >>> len(_cache_key(f))  # doctest: +SKIP
    64
    """
    digest_input = inspect.getsource(func) + "\n" + _toolchain_fingerprint()
    return hashlib.sha256(digest_input.encode()).hexdigest()


def create_target_machine() -> llvm_binding.TargetMachine:
    """
    Create an `llvmlite` `TargetMachine` matching Numba's CPU JIT settings.

    Uses `codemodel="jitdefault"` and `jit=True` across Linux (`x86_64`,
    `aarch64`) and macOS (`arm64`, `x86_64`) so MCJIT relocations against
    in-process runtime helpers (`libm`, `NRT_*`) succeed regardless of
    virtual-address distance.

    Returns
    -------
    llvmlite.binding.TargetMachine
        Configured host `TargetMachine` for MCJIT object emission and loading.

    See Also
    --------
    build : Uses this `TargetMachine` to emit `kernel.o`.
    numba_enzyme.runtime.load : Uses this `TargetMachine` to construct MCJIT.

    Examples
    --------
    >>> from numba_enzyme.build import create_target_machine
    >>> bool(create_target_machine().triple)
    True
    """
    llvm_binding.initialize_all_targets()
    llvm_binding.initialize_all_asmprinters()

    triple = llvm_binding.get_process_triple()
    target = llvm_binding.Target.from_triple(triple)
    arch = triple.split("-", 1)[0]
    if arch.startswith("x86"):
        reloc = "static"
    elif arch.startswith("ppc"):
        reloc = "pic"
    else:
        reloc = "default"

    try:
        features = llvm_binding.get_host_cpu_features().flatten()
    except RuntimeError:
        # `get_host_cpu_features()` raises RuntimeError on Apple Silicon;
        # `get_host_cpu_name()` (e.g. "apple-m1") already encodes the ISA.
        features = ""

    return target.create_target_machine(
        cpu=llvm_binding.get_host_cpu_name(),
        features=features,
        opt=2,
        reloc=reloc,
        codemodel="jitdefault",
        jit=True,
    )


@lru_cache(maxsize=4)
def _load_enzyme_ffi(lib_path: Path) -> ctypes.CDLL:
    """
    Load `libNumbaEnzyme` via `ctypes` and configure its C ABI prototypes.

    Parameters
    ----------
    lib_path : pathlib.Path
        Resolved path to `libNumbaEnzyme`.

    Returns
    -------
    ctypes.CDLL
        Loaded library handle with `numba_enzyme_compile_ir` and
        `numba_enzyme_free_string` prototypes configured.
    """
    lib = ctypes.CDLL(str(lib_path))
    lib.numba_enzyme_compile_ir.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.numba_enzyme_compile_ir.restype = ctypes.c_int
    lib.numba_enzyme_free_string.argtypes = [ctypes.c_void_p]
    lib.numba_enzyme_free_string.restype = None
    return lib


def _run_enzyme_pass(
    lib_path: Path, kernel_ir: str, entry_symbol: str, driver_ir: str
) -> tuple[str, str]:
    """
    Extract, link, and differentiate a kernel in-process via `libNumbaEnzyme`.

    Parameters
    ----------
    lib_path : pathlib.Path
        Path to `libNumbaEnzyme`.
    kernel_ir : str
        Full LLVM IR module emitted by Numba.
    entry_symbol : str
        Mangled kernel symbol to extract and differentiate.
    driver_ir : str
        Synthesised Enzyme driver LLVM IR module.

    Returns
    -------
    combined_ir : str
        Post-extraction, post-link LLVM IR before Enzyme runs.
    enzyme_ir : str
        Differentiated LLVM IR produced by Enzyme.

    Raises
    ------
    RuntimeError
        If extraction, module linking, Enzyme differentiation, or LLVM IR
        verification fails.
    """
    ffi = _load_enzyme_ffi(lib_path)
    combined_ptr = ctypes.c_void_p()
    enzyme_ptr = ctypes.c_void_p()
    err_ptr = ctypes.c_void_p()
    try:
        rc = ffi.numba_enzyme_compile_ir(
            kernel_ir.encode("utf-8"),
            entry_symbol.encode("utf-8"),
            driver_ir.encode("utf-8"),
            ctypes.byref(combined_ptr),
            ctypes.byref(enzyme_ptr),
            ctypes.byref(err_ptr),
        )
        if rc != 0:
            err_msg = (
                ctypes.string_at(err_ptr.value).decode("utf-8", errors="replace")
                if err_ptr.value
                else "unknown Enzyme compilation failure"
            )
            raise RuntimeError(f"Enzyme compilation failed: {err_msg}")
        combined_ir = (
            ctypes.string_at(combined_ptr.value).decode("utf-8")
            if combined_ptr.value
            else ""
        )
        enzyme_ir = ctypes.string_at(enzyme_ptr.value).decode("utf-8")
        return combined_ir, enzyme_ir
    finally:
        if combined_ptr.value:
            ffi.numba_enzyme_free_string(combined_ptr)
        if enzyme_ptr.value:
            ffi.numba_enzyme_free_string(enzyme_ptr)
        if err_ptr.value:
            ffi.numba_enzyme_free_string(err_ptr)


def _compile_object(enzyme_ir: str) -> bytes:
    """
    Optimize differentiated LLVM IR at `-O2` and emit a native object file.

    Parameters
    ----------
    enzyme_ir : str
        Differentiated LLVM IR text produced by Enzyme.

    Returns
    -------
    bytes
        Relocatable native object bytes ready for `llvmlite` MCJIT loading.
    """
    tm = create_target_machine()
    mod = llvm_binding.parse_assembly(enzyme_ir)
    mod.triple = tm.triple
    mod.verify()

    pto = llvm_binding.create_pipeline_tuning_options(speed_level=2)
    pb = llvm_binding.create_pass_builder(tm, pto)
    mpm = pb.getModulePassManager()
    mpm.run(mod, pb)

    return tm.emit_object(mod)


def build(func: Callable) -> BuiltKernel:
    """
    Build, or fetch from cache, the differentiated native object for `func`.

    Lowers `func` with Numba, synthesises its Enzyme driver, extracts and
    differentiates the linked IR in-process via `libNumbaEnzyme`, optimizes
    and emits a native relocatable object via `llvmlite`, and caches the
    result on disk. If an identical build (same source text and toolchain
    fingerprint) is already cached on disk, the cached object is returned
    directly without recompiling.

    Parameters
    ----------
    func : callable
        A Python function whose parameters and return value are each
        annotated with a `numba_enzyme.types` class.

    Returns
    -------
    BuiltKernel
        The compiled (or cached) native object and its entry-point names.

    See Also
    --------
    BuiltKernel : The result this function returns.
    numba_enzyme.runtime.load : Loads the result into Python callables.

    Examples
    --------
    >>> from numba_enzyme.build import build
    >>> from numba_enzyme.types import Float64
    >>> def f(x: Float64) -> Float64:
    ...     return x * x
    >>> build(f).path.suffix  # doctest: +SKIP
    '.o'
    """
    entry_dir = _cache_dir() / _cache_key(func)
    obj_path = entry_dir / "kernel.o"
    meta_path = entry_dir / "meta.json"

    if obj_path.is_file() and meta_path.is_file():
        # Lowering is skipped on cache hits because Numba embeds an internal
        # per-qualname version counter in mangled symbol names that increments
        # on repeated `nb.cfunc` compilations in the same process. Persisting
        # the symbol names in `meta.json` keeps them aligned with `kernel.o`.
        raw_meta = json.loads(meta_path.read_text())
        try:
            meta = {
                "grad_symbol": raw_meta["grad_symbol"],
                "jvp_symbol": raw_meta["jvp_symbol"],
                "n_args": raw_meta["n_args"],
                "arg_specs": tuple(ArgSpec(**spec) for spec in raw_meta["arg_specs"]),
            }
        except (KeyError, TypeError):
            pass
        else:
            return BuiltKernel(path=obj_path, from_cache=True, **meta)

    tc = get_toolchain()
    kernel = lower(func)
    drv = synthesise(kernel)
    combined_ir, enzyme_ir = _run_enzyme_pass(
        tc.enzyme_lib, kernel.ir, kernel.entry_symbol, drv.ir
    )
    obj_bytes = _compile_object(enzyme_ir)

    entry_dir.mkdir(parents=True, exist_ok=True)
    (entry_dir / "kernel_full.ll").write_text(kernel.ir)
    (entry_dir / "driver.ll").write_text(drv.ir)
    (entry_dir / "combined.ll").write_text(combined_ir)
    (entry_dir / "enzyme_out.ll").write_text(enzyme_ir)
    obj_path.write_bytes(obj_bytes)

    meta = {
        "grad_symbol": drv.grad_symbol,
        "jvp_symbol": drv.jvp_symbol,
        "n_args": kernel.n_args,
        "arg_specs": kernel.arg_specs,
    }
    meta_path.write_text(json.dumps(meta, default=vars))

    return BuiltKernel(path=obj_path, from_cache=False, **meta)
