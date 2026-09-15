"""
Orchestrate and cache the full lowering-to-shared-object pipeline.

Wires lowering, driver synthesis, ``llvm-link``, the Enzyme ``opt``
pass, and the final shared-object compile into a single call, and
caches built ``.so`` files on disk, keyed on the function's source text
and a fingerprint of the toolchain used to build it.

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

import hashlib
import inspect
import json
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from numba_enzyme.driver import synthesise
from numba_enzyme.lowering import lower
from numba_enzyme.toolchain import get_toolchain

_CACHE_DIR_ENV_VAR = "NUMBA_ENZYME_CACHE_DIR"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "numba_enzyme"
_CPU_CACHE_SCHEMA = 3


@dataclass(frozen=True)
class BuiltKernel:
    """
    A compiled, cacheable shared object exposing derivative entry points.

    Attributes
    ----------
    path : pathlib.Path
        Path to the built ``.so`` file.
    jvp_symbol : str
        Name of the forward-mode entry point exported by `path`.
    vjp_symbol : str
        Name of the reverse-mode vector-Jacobian product entry point exported
        by `path`.
    n_args : int
        Number of scalar arguments the original function takes.
    n_outputs : int
        Number of scalar output components.
    arg_types : tuple of str
        LLVM textual type of each scalar input.
    return_type : str
        LLVM textual type of each output component.
    from_cache : bool
        Whether this result was served from the on-disk cache rather
        than freshly compiled.

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
    jvp_symbol: str
    vjp_symbol: str
    n_args: int
    n_outputs: int
    arg_types: tuple[str, ...]
    return_type: str
    from_cache: bool


def _cache_dir() -> Path:
    """
    Return the directory `build` caches compiled shared objects in.

    Defaults to ``~/.cache/numba_enzyme``, overridable via the
    ``NUMBA_ENZYME_CACHE_DIR`` environment variable.

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
    Fingerprint the resolved toolchain by each tool's mtime and size.

    Used as part of the cache key so rebuilding any tool -- most
    notably the Enzyme plugin itself -- invalidates every cache entry
    automatically.

    Returns
    -------
    str
        A string encoding each tool's path, modification time, and
        size.

    See Also
    --------
    numba_enzyme.toolchain.get_toolchain : Resolves the tools fingerprinted
        here.

    Examples
    --------
    >>> from numba_enzyme.build import _toolchain_fingerprint
    >>> _toolchain_fingerprint()  # doctest: +SKIP
    '/usr/bin/clang-15:...|/usr/bin/llvm-link-15:...'
    """
    tc = get_toolchain()
    parts = []
    for path in (tc.clang, tc.llvm_link, tc.opt, tc.enzyme_plugin):
        stat = path.stat()
        parts.append(f"{path}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts)


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
    digest_input = (
        f"cpu-cache-schema={_CPU_CACHE_SCHEMA}\n"
        + inspect.getsource(func)
        + "\n"
        + _toolchain_fingerprint()
    )
    return hashlib.sha256(digest_input.encode()).hexdigest()


_LIBC_MAP_RE = re.compile(r"(/\S*/lib(c|m)\.so(?:\.\d+)*)$")


def _locate_runtime_libs() -> list[Path]:
    """
    Find the running process's own loaded libc/libm, via ``/proc/self/maps``.

    Used only for the vendored-crt link path (see `build`): a vendored
    toolchain ships its own crt startup objects and a static `libgcc`,
    but deliberately does not vendor libc/libm themselves. Every Linux
    target that can run this Python process at all has *already*
    loaded some libc into it by the time this runs, so reading its own
    memory map back is a distro/layout-agnostic way to find the exact
    file to link against -- unlike hardcoding a path convention such
    as Debian's ``/lib/x86_64-linux-gnu/``, which doesn't exist on e.g.
    Fedora/RHEL/Arch. For instance a link built this way on
    Debian trixie produces a shared object whose only `NEEDED` entries
    are ``libc.so.6``/``libm.so.6``.

    Returns
    -------
    list of pathlib.Path
        Absolute paths to the loaded libc and (if present as a
        separate mapping as recent glibc merges it into libc) libm.

    See Also
    --------
    build : Used when a vendored crt directory exists.

    Examples
    --------
    >>> from numba_enzyme.build import _locate_runtime_libs
    >>> any(p.name.startswith("libc.so") for p in _locate_runtime_libs())
    True
    """
    found: dict[str, Path] = {}
    with open("/proc/self/maps") as f:
        for line in f:
            match = _LIBC_MAP_RE.search(line)
            if match:
                found.setdefault(match.group(2), Path(match.group(1)))
    return list(found.values())


def build(func: Callable) -> BuiltKernel:
    """
    Build, or fetch from cache, the shared object for a function.

    Lowers `func` with Numba, synthesises its Enzyme driver, links the
    two with ``llvm-link``, runs the standalone Enzyme ``opt`` pass, and
    compiles the result to a shared object with ``clang``. If an
    identical build (same source text and toolchain fingerprint) is
    already cached on disk, the cached one is returned without any
    subprocess calls.

    Parameters
    ----------
    func : callable
        A Python function whose parameters and return value are each
        annotated with a `numba_enzyme.types` class.

    Returns
    -------
    BuiltKernel
        The compiled (or cached) shared object and its entry-point
        names.

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
    '.so'
    """
    entry_dir = _cache_dir() / _cache_key(func)
    so_path = entry_dir / "kernel.so"
    meta_path = entry_dir / "meta.json"

    if so_path.is_file() and meta_path.is_file():
        # if the statement is true, lower() isn't call again here. The
        # reason is that Numba embeds an internal version counter in the
        # mangled symbol name that increments every time nb.cfunc compiles
        # the same function again in the same process (e.g. ...B2v1...
        # vs ...B2v2...), even with identical source. Recomputing symbol
        # names on a cache hit would silently drift from what's actually
        # embedded into the already-built .so. Persist them from the
        # original build instead.
        meta = json.loads(meta_path.read_text())
        meta["arg_types"] = tuple(meta["arg_types"])
        return BuiltKernel(path=so_path, from_cache=True, **meta)

    kernel = lower(func)
    drv = synthesise(kernel)
    tc = get_toolchain()

    entry_dir.mkdir(parents=True, exist_ok=True)
    kernel_ll = entry_dir / "kernel.ll"
    driver_ll = entry_dir / "driver.ll"
    combined_ll = entry_dir / "combined.ll"
    enzyme_out_ll = entry_dir / "enzyme_out.ll"

    kernel_ll.write_text(kernel.ir)
    driver_ll.write_text(drv.ir)

    subprocess.run(
        [
            str(tc.llvm_link),
            str(kernel_ll),
            str(driver_ll),
            "-S",
            "-o",
            str(combined_ll),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(tc.opt),
            f"-load-pass-plugin={tc.enzyme_plugin}",
            "-passes=enzyme",
            "-S",
            str(combined_ll),
            "-o",
            str(enzyme_out_ll),
        ],
        check=True,
    )
    # `clang -shared` shells out to a separate linker executable for the
    # final link step. A vendored toolchain (see toolchain.py) ships its
    # own `ld.lld` right next to clang specifically so this works with no
    # system linker present at all. The system/dev-mode toolchain doesn't
    # need this, a normal dev machine already has a system `ld`. So only
    # pass it when the vendored linker actually exists next to clang.
    vendored_lld = tc.clang.parent / "ld.lld"
    extra_link_args = [f"-fuse-ld={vendored_lld}"] if vendored_lld.is_file() else []

    # Even with a linker present, `-shared` needs C runtime startup
    # objects (crti.o/crtn.o/crtbeginS.o/crtendS.o) and libgcc, normally
    # supplied by a system C toolchain (gcc/libc6-dev). A vendored
    # toolchain ships these too (see cibw_before_all.sh); `-nostdlib`
    # stops clang from also trying to supply its own defaults (which
    # would look in system paths that don't exist here) so every input
    # is explicit. libc/libm are deliberately NOT vendored -- they're
    # located on the actual target machine instead (`_locate_runtime_libs`),
    # since baking in one distro's copy or path convention would break
    # on any other. Dev-mode/system toolchain untouched: this whole
    # block only runs when `_vendor/crt/` actually exists.
    vendored_crt_dir = tc.clang.parent.parent / "crt"
    post_input_args = []
    if vendored_crt_dir.is_dir():
        extra_link_args.append("-nostdlib")
        post_input_args = [
            "-x",
            "none",
            str(vendored_crt_dir / "crti.o"),
            str(vendored_crt_dir / "crtbeginS.o"),
            str(vendored_crt_dir / "libgcc.a"),
            *[str(p) for p in _locate_runtime_libs()],
            str(vendored_crt_dir / "crtendS.o"),
            str(vendored_crt_dir / "crtn.o"),
        ]

    # TODO: consider `-O3` at some point
    subprocess.run(
        [
            str(tc.clang),
            "-O2",
            "-fPIC",
            "-shared",
            *extra_link_args,
            "-x",
            "ir",
            str(enzyme_out_ll),
            *post_input_args,
            "-o",
            str(so_path),
        ],
        check=True,
    )

    meta = {
        "jvp_symbol": drv.jvp_symbol,
        "vjp_symbol": drv.vjp_symbol,
        "n_args": kernel.n_args,
        "n_outputs": kernel.n_outputs,
        "arg_types": kernel.arg_types[2:],
        "return_type": kernel.return_type,
    }
    meta_path.write_text(json.dumps(meta))

    return BuiltKernel(path=so_path, from_cache=False, **meta)
