"""
Differentiate Numba-CUDA-MLIR device functions with Enzyme.

The CUDA backend lowers one concrete ``numba-cuda-mlir`` device-function
specialization through MLIR to NVVM-flavoured LLVM IR, adds small C-ABI
forward- and reverse-mode entry points, runs Enzyme before libNVVM, and links the
resulting LTO IR through ``cuda.declare_device``. The public transforms return
placeholders that specialize from the concrete argument types at each call site
inside another MLIR-backed ``@cuda.jit`` function.

All CUDA imports are lazy so CPU-only installations continue to work without
the optional ``numba-cuda-mlir`` dependency.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import subprocess
import threading
import typing
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import llvmlite
import numba as nb
from llvmlite import ir

from numba_enzyme.build import _cache_dir, _toolchain_fingerprint
from numba_enzyme.toolchain import get_toolchain

_CUDA_CC_ENV_VAR = "NUMBA_ENZYME_CUDA_CC"
_CUDA_CACHE_SCHEMA = 7
_LOADED_CUDA_DERIVATIVES = {}
_LOAD_LOCK = threading.RLock()
_MLIR_COMPILE_LOCK = threading.RLock()

_SCALAR_IR_TYPE = {"float32": ir.FloatType(), "float64": ir.DoubleType()}


class CUDAEnzymeError(RuntimeError):
    """Raised when a CUDA derivative cannot be compiled or linked."""


@dataclass(frozen=True)
class CUDALoweredKernel:
    """A concrete Numba-CUDA-MLIR device-function specialization."""

    ir: str
    entry_symbol: str
    arg_types: tuple[object, ...]
    return_type: object
    compute_capability: tuple[int, int]
    fastmath: bool


@dataclass(frozen=True)
class CUDASynthesisedDriver:
    """
    LLVM driver module containing the requested CUDA entry points.

    Attributes
    ----------
    ir : str
        The driver module's full LLVM IR text.
    modes : frozenset
        Derivative modes the module defines entry points for.
    symbols : dict
        External symbol names for each requested mode. A tuple-returning primal
        has one symbol per mode; a scalar one has a symbol per primal argument
        wherever the public shape is a tuple, since the backend cannot return
        an aggregate across an ``abi="c"`` boundary.

    See Also
    --------
    synthesise_cuda : Builds a `CUDASynthesisedDriver` instance.
    """

    ir: str
    modes: frozenset
    symbols: dict


@dataclass(frozen=True)
class CUDABuiltKernel:
    """
    Differentiated device code and the Numba signatures of its entry points.

    `code` is NVVM LTO IR, not PTX: numba-cuda-mlir compiles the calling kernel
    to LTO IR as well whenever a link item is LTO IR, which lets nvJitLink
    inline the derivative into its caller rather than leaving an opaque call
    carrying a parameter per primal argument.

    Attributes
    ----------
    code : bytes
        NVVM LTO IR holding the differentiated entry points.
    modes : frozenset
        Derivative modes that were built.
    symbols : dict
        External symbol names for each built mode.
    signatures : dict
        Public Numba device-call signature of each mode the primal's shape
        supports, whether or not it was built.
    n_args : int
        Number of scalar primal arguments.
    shape : str
        The primal's shape, as `_primal_shape` reports it.
    path : pathlib.Path
        Location of `code` in the on-disk cache, used as the link item.
    from_cache : bool
        Whether this result was served from that cache rather than built.

    See Also
    --------
    build_cuda : Builds a `CUDABuiltKernel` instance.
    """

    code: bytes
    modes: frozenset
    symbols: dict
    signatures: dict
    n_args: int
    shape: str
    path: Path
    from_cache: bool
    depth: int = 1
    n_dirs: int = 1


@dataclass(frozen=True)
class CUDADifferentiable:
    """
    Compiled CUDA derivatives, ready to be compiled in at a call site.

    Attributes
    ----------
    implementations : dict
        Python implementation of each built mode. Each takes the parameters of
        `lazy_cuda_derivative`'s placeholder for that mode and calls the
        linked entry points directly, so the MLIR frontend compiles it in place
        of the placeholder.
    n_args : int
        Number of scalar primal arguments.

    See Also
    --------
    differentiate_cuda : Builds a `CUDADifferentiable` instance.
    """

    implementations: dict
    n_args: int
    # The external declarations behind the tuple implementations, keyed by
    # mode. Calling one directly keeps a kernel's LTO IR identical across
    # processes, which an implementation cannot when the call is too wide to
    # inline: Numba-CUDA-MLIR names an out-of-line overload after the
    # dispatcher's id(). Empty for a scalar primal, whose public shapes are
    # tuples the C ABI cannot carry.
    externals: dict | None = None


def is_cuda_device_function(func) -> bool:
    """
    Return whether a function is a CUDA device dispatcher.

    Parameters
    ----------
    func : object
        Object to inspect.

    Returns
    -------
    bool
        Whether `func` is a real ``@cuda.jit(device=True)`` dispatcher.
    """

    targetoptions = getattr(func, "targetoptions", None)
    if not isinstance(targetoptions, dict) or not targetoptions.get("device"):
        return False
    cls = type(func)
    return cls.__name__ == "MLIRDispatcher" and cls.__module__.startswith(
        "numba_cuda_mlir"
    )


def _cuda_imports():
    """
    Import the Numba-CUDA-MLIR interfaces needed by this backend.

    Returns
    -------
    tuple
        CUDA module, its type module, compiler helpers, and lowering modules.

    Raises
    ------
    CUDAEnzymeError
        If a sufficiently recent Numba-CUDA-MLIR package is unavailable.
    """

    try:
        from numba_cuda_mlir import cuda, tools, types
        from numba_cuda_mlir._mlir import ir as mlir_ir
        from numba_cuda_mlir._mlir.passmanager import PassManager
        from numba_cuda_mlir.lowering_utilities import context as mlir_context
        from numba_cuda_mlir.mlir_optimization import (
            _call_llvm70_capi,
            _compile_to_ltoir,
            _compile_to_ptx,
            _needs_llvm70_path,
            _nvvm_options,
            _prepare_llvm_ir,
            get_base_pipeline,
        )
        from numba_cuda_mlir.numba_cuda.cudadrv.nvvm import LibDevice
        from numba_cuda_mlir.optimization import run_pre_codegen_patterns
    except (ImportError, AttributeError) as exc:
        raise CUDAEnzymeError(
            "CUDA differentiation requires numba-cuda-mlir >= 0.5.1; "
            "install numba-enzyme-cuda[cuda]"
        ) from exc
    return {
        "cuda": cuda,
        "types": types,
        "tools": tools,
        "mlir_ir": mlir_ir,
        "PassManager": PassManager,
        "mlir_context": mlir_context,
        "compile_to_ltoir": _compile_to_ltoir,
        "compile_to_ptx": _compile_to_ptx,
        "llvm70_ir": _call_llvm70_capi,
        "needs_llvm70": _needs_llvm70_path,
        "nvvm_options": _nvvm_options,
        "prepare_llvm_ir": _prepare_llvm_ir,
        "base_pipeline": get_base_pipeline,
        "LibDevice": LibDevice,
        "pre_codegen": run_pre_codegen_patterns,
    }


def _annotation_type(annotation, cuda_types=None):
    """
    Convert one Python annotation to a Numba type.

    Parameters
    ----------
    annotation : object
        Numba type or callable `numba_enzyme.types` annotation.
    cuda_types : module, optional
        Numba-CUDA-MLIR type module, imported lazily when omitted.

    Returns
    -------
    numba_cuda_mlir.types.Type
        Normalized Numba-CUDA-MLIR type.

    Raises
    ------
    CUDAEnzymeError
        If the annotation cannot produce a Numba type.
    """
    if cuda_types is None:
        cuda_types = _cuda_imports()["types"]
    if isinstance(annotation, cuda_types.Type):
        return annotation
    type_by_name = {
        "float32": cuda_types.float32,
        "float64": cuda_types.float64,
        "int32": cuda_types.int32,
        "int64": cuda_types.int64,
    }
    converted = type_by_name.get(str(annotation))
    if converted is not None:
        return converted
    try:
        result = annotation()
    except TypeError as exc:
        raise CUDAEnzymeError(
            f"annotation {annotation!r} is not a Numba or numba_enzyme type"
        ) from exc
    converted = type_by_name.get(str(result))
    if converted is None:
        raise CUDAEnzymeError(
            f"annotation {annotation!r} did not produce a Numba-CUDA-MLIR type"
        )
    return converted


def _normalise_signature(func, signature=None):
    """
    Return the types for one CUDA specialization.

    Parameters
    ----------
    func : numba_cuda_mlir.descriptor.MLIRDispatcher
        Device dispatcher whose Python function supplies annotations.
    signature : numba.core.typing.Signature, optional
        Explicit specialization, overriding annotations.

    Returns
    -------
    tuple
        Tuple containing the argument-type tuple and return type.

    Raises
    ------
    CUDAEnzymeError
        If neither a complete signature nor complete annotations are present.
    """

    pyfunc = func.py_func
    cuda_types = _cuda_imports()["types"]
    if signature is not None:
        from numba_cuda_mlir.numba_cuda.core import sigutils

        arg_types, return_type = sigutils.normalize_signature(signature)
        if return_type is None:
            raise CUDAEnzymeError(
                "the CUDA derivative signature must include a return type"
            )
        return tuple(
            _annotation_type(t, cuda_types) for t in arg_types
        ), _annotation_type(return_type, cuda_types)

    hints = typing.get_type_hints(pyfunc)
    params = inspect.signature(pyfunc).parameters
    try:
        arg_types = tuple(_annotation_type(hints[name], cuda_types) for name in params)
        return_type = _annotation_type(hints["return"], cuda_types)
    except KeyError as exc:
        raise CUDAEnzymeError(
            f"{pyfunc!r} is missing a type annotation for {exc.args[0]!r}; "
            "pass signature=<return type>(<argument types>) to the derivative transform"
        ) from exc
    return arg_types, return_type


def _validate_signature(arg_types, return_type):
    """
    Validate a signature against the initial CUDA backend's scalar subset.

    Parameters
    ----------
    arg_types : tuple of numba.types.Type
        Argument types to validate.
    return_type : numba.types.Type
        Return type to validate.

    Raises
    ------
    CUDAEnzymeError
        If the signature is empty, non-scalar, or heterogeneous.
    """
    if not arg_types:
        raise CUDAEnzymeError("CUDA differentiation requires at least one argument")
    cuda_types = _cuda_imports()["types"]
    if return_type in (cuda_types.void, cuda_types.none):
        raise CUDAEnzymeError(
            "a CUDA primal has to return its outputs, as one scalar or as a "
            "homogeneous tuple; writing them through an output array is not "
            "supported"
        )
    supported = set(_SCALAR_IR_TYPE)
    return_name = str(return_type)
    arg_names = tuple(map(str, arg_types))
    if return_name not in supported or any(t not in supported for t in arg_names):
        names = ", ".join(sorted(supported))
        raise CUDAEnzymeError(
            f"the CUDA backend currently supports only scalar {names} arguments "
            "and returns"
        )
    if any(t != return_name for t in arg_names):
        raise CUDAEnzymeError(
            "the initial CUDA backend requires every argument to have the same "
            "floating-point type as the return value"
        )


def _primal_shape(return_type):
    """
    Classify a primal by its return type.

    Parameters
    ----------
    return_type : numba.types.Type
        The primal's return type.

    Returns
    -------
    str
        ``"tuple"`` for a primal returning a tuple, ``"scalar"`` otherwise.
        A primal returning nothing is classified as ``"scalar"`` so that
        `_validate_signature` is the single place that rejects it.
    """
    cuda_types = _cuda_imports()["types"]
    if isinstance(return_type, cuda_types.BaseTuple):
        return "tuple"
    return "scalar"


def _argument_widths(arg_types):
    """
    Describe each primal argument as a scalar or a tuple of scalars.

    Numba-CUDA-MLIR flattens a tuple argument into one scalar parameter per
    element under the C ABI, so Enzyme sees scalars either way. The widths are
    what lets an entry point take the tuple as an array and reconstitute those
    scalars itself, rather than making the caller spell out every one.

    Parameters
    ----------
    arg_types : tuple of numba.types.Type
        Primal argument types.

    Returns
    -------
    tuple
        One entry per argument: `None` for a scalar, otherwise the number of
        elements in the tuple.
    """
    cuda_types = _cuda_imports()["types"]
    return tuple(
        t.count if isinstance(t, cuda_types.UniTuple) else None for t in arg_types
    )


def _flat_argument_count(arg_types) -> int:
    """Return how many scalars a primal's arguments flatten to."""
    return sum(1 if width is None else width for width in _argument_widths(arg_types))


def _validate_tuple_signature(arg_types, return_type):
    """
    Validate a tuple-returning primal's signature.

    The shape is ``UniTuple(dtype, n)(x0, ..., xn)``: a homogeneous tuple
    return, which Numba-CUDA-MLIR lowers to an LLVM struct returned by value
    and Enzyme differentiates directly. Each argument is a scalar of the same
    floating-point type, or a homogeneous tuple of them -- Numba-CUDA-MLIR
    flattens a tuple argument to one scalar parameter per element, so Enzyme
    sees the same flat signature either way.

    Parameters
    ----------
    arg_types : tuple of numba.types.Type
        Argument types to validate.
    return_type : numba.types.Type
        Return type to validate.

    Raises
    ------
    CUDAEnzymeError
        If the signature does not have that shape.
    """
    cuda_types = _cuda_imports()["types"]
    if not arg_types:
        raise CUDAEnzymeError("CUDA differentiation requires at least one argument")
    if not isinstance(return_type, cuda_types.UniTuple) or return_type.count < 1:
        raise CUDAEnzymeError(
            "a tuple-returning primal must return a non-empty homogeneous tuple, "
            f"not {return_type}"
        )
    supported = set(_SCALAR_IR_TYPE)
    dtype = str(return_type.dtype)
    leaves = []
    for argument in arg_types:
        if isinstance(argument, cuda_types.UniTuple):
            if argument.count < 1:
                raise CUDAEnzymeError("a tuple argument must have at least one element")
            leaves.append(str(argument.dtype))
        else:
            leaves.append(str(argument))
    if dtype not in supported or any(leaf != dtype for leaf in leaves):
        names = ", ".join(sorted(supported))
        raise CUDAEnzymeError(
            f"a tuple-returning primal requires scalar {names} arguments, or "
            "homogeneous tuples of them, matching its tuple's element type"
        )


def _validate_primal_signature(shape, arg_types, return_type):
    """
    Validate a primal's signature against its shape.

    Parameters
    ----------
    shape : str
        The primal's shape, as `_primal_shape` reports it.
    arg_types : tuple of numba.types.Type
        Argument types to validate.
    return_type : numba.types.Type
        Return type to validate.
    """
    {"scalar": _validate_signature, "tuple": _validate_tuple_signature}[shape](
        arg_types, return_type
    )


def _parse_compute_capability(value: str) -> tuple[int, int]:
    """
    Parse a CUDA compute-capability string.

    Parameters
    ----------
    value : str
        Value such as ``"8.0"``, ``"80"``, or ``"sm_80"``.

    Returns
    -------
    tuple of int
        The ``(major, minor)`` pair.

    Raises
    ------
    CUDAEnzymeError
        If `value` does not identify a compute capability.
    """
    text = value.lower().removeprefix("sm_").removeprefix("compute_")
    if "." in text:
        pieces = text.split(".")
    elif len(text) == 2 and text.isdigit():
        pieces = list(text)
    else:
        pieces = []
    if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
        raise CUDAEnzymeError(
            f"invalid compute capability {value!r}; use e.g. '8.0' or '80'"
        )
    return int(pieces[0]), int(pieces[1])


def _compute_capability(cc, tools) -> tuple[int, int]:
    """
    Resolve the compute capability without initializing the CUDA driver.

    Parameters
    ----------
    cc : tuple of int or str or None
        Explicit compute capability, if supplied.
    tools : module
        Numba-CUDA-MLIR target-discovery helpers.

    Returns
    -------
    tuple of int
        Resolved ``(major, minor)`` pair.
    """
    if cc is not None:
        if isinstance(cc, str):
            result = _parse_compute_capability(cc)
            return _validate_compute_capability(result)
        if len(cc) != 2:
            raise CUDAEnzymeError("compute capability must be a (major, minor) pair")
        result = int(cc[0]), int(cc[1])
        return _validate_compute_capability(result)
    override = os.environ.get(_CUDA_CC_ENV_VAR)
    if override:
        return _validate_compute_capability(_parse_compute_capability(override))
    return _validate_compute_capability(tuple(tools.get_gpu_compute_capability(tuple)))


def _validate_compute_capability(cc):
    """
    Validate the GPU architecture supported by the LLVM 15 bridge.

    Parameters
    ----------
    cc : tuple of int
        Compute capability to validate.

    Returns
    -------
    tuple of int
        The validated compute capability.

    Raises
    ------
    CUDAEnzymeError
        If the architecture needs an older or newer compiler path.
    """

    if cc < (7, 0):
        raise CUDAEnzymeError("Numba-CUDA-MLIR requires compute capability 7.0+")
    if cc >= (10, 0):
        raise CUDAEnzymeError(
            "CUDA differentiation currently supports compute capabilities 7.x-9.x; "
            "Blackwell's LLVM 20 NVVM path is incompatible with the bundled LLVM 15 "
            "Enzyme plugin"
        )
    return cc


@contextmanager
def _mlir_target_override(tools, compute_capability):
    """
    Let explicit-target lowering run without requiring a visible GPU.

    Parameters
    ----------
    tools : module
        Numba-CUDA-MLIR target-discovery helpers.
    compute_capability : tuple of int
        Explicit target used while lowering.

    Yields
    ------
    None
        Control while the temporary target override is active.
    """

    with _MLIR_COMPILE_LOCK:
        previous = tools._cached_cc
        tools._cached_cc = compute_capability
        try:
            yield
        finally:
            tools._cached_cc = previous


def lower_cuda(func, signature=None, cc=None) -> CUDALoweredKernel:
    """
    Lower a CUDA device dispatcher to one concrete NVVM IR specialization.

    Parameters
    ----------
    func : numba_cuda_mlir.descriptor.MLIRDispatcher
        Device function to lower.
    signature : numba_cuda_mlir.typing.Signature, optional
        Concrete specialization, otherwise inferred from annotations.
    cc : tuple of int or str, optional
        CUDA compute capability.

    Returns
    -------
    CUDALoweredKernel
        Lowered NVVM IR and its ABI metadata.

    Raises
    ------
    CUDAEnzymeError
        If `func` is not a supported CUDA device function.
    """

    if not is_cuda_device_function(func):
        raise CUDAEnzymeError(
            "expected a function decorated with @cuda.jit(device=True)"
        )

    backend = _cuda_imports()
    arg_types, return_type = _normalise_signature(func, signature)
    _validate_primal_signature(_primal_shape(return_type), arg_types, return_type)
    compute_capability = _compute_capability(cc, backend["tools"])

    options = getattr(func, "targetoptions", {})
    fastmath = bool(options.get("fastmath", False))
    cuda_signature = return_type(*arg_types)
    compile_options = {
        "device": True,
        "abi": "c",
        "cc": compute_capability,
        "fastmath": fastmath,
        "debug": bool(options.get("debug", False)),
        "lineinfo": bool(options.get("lineinfo", False)),
        "opt": options.get("opt"),
    }
    compile_options = {k: v for k, v in compile_options.items() if v is not None}

    # compile_mlir(..., optimized=False) is public. The final translation
    # helpers are private in 0.5.x because the package exposes PTX, not its
    # pre-libNVVM LLVM handoff. Keeping the bridge here makes that dependency
    # explicit and easy to replace when a public hook is added upstream.
    with _mlir_target_override(backend["tools"], compute_capability):
        mlir_text = backend["cuda"].compile_mlir(
            func.py_func, cuda_signature, optimized=False, **compile_options
        )
        with backend["mlir_context"].get_context():
            module = backend["mlir_ir"].Module.parse(mlir_text)
            pass_manager = backend["PassManager"].parse(backend["base_pipeline"]())
            pass_manager.run(module.operation)
            backend["pre_codegen"](module)
            chip = f"sm_{compute_capability[0]}{compute_capability[1]}"
            if backend["needs_llvm70"](chip):
                llvm_bytes = backend["llvm70_ir"](
                    module,
                    {
                        "chip": chip,
                        "opt_level": 0 if options.get("opt") is False else 3,
                        "debug": compile_options["debug"],
                        "lineinfo": compile_options["lineinfo"],
                    },
                    gen_llvmir=True,
                )
            else:
                llvm_bytes = backend["prepare_llvm_ir"](
                    module,
                    preserve_debug_info=compile_options["debug"]
                    or compile_options["lineinfo"],
                )

    # Numba-CUDA-MLIR mangles the qualified name, which for a nested or
    # generated function is not its __name__: a closure's primal would
    # otherwise be looked up under a symbol the module never defines.
    entry_symbol = backend["tools"].generate_mangled_name(
        func.py_func.__qualname__, arg_types
    )
    return CUDALoweredKernel(
        ir=llvm_bytes.decode(),
        entry_symbol=entry_symbol,
        arg_types=arg_types,
        return_type=return_type,
        compute_capability=compute_capability,
        fastmath=fastmath,
    )


def _target_lines(kernel_ir: str) -> tuple[str, str]:
    """
    Extract and validate the target triple and data layout.

    Parameters
    ----------
    kernel_ir : str
        Numba-CUDA-MLIR LLVM module.

    Returns
    -------
    tuple of str
        NVPTX target triple and data layout.

    Raises
    ------
    CUDAEnzymeError
        If the module does not target 64-bit NVIDIA PTX.
    """
    triple = datalayout = None
    for line in kernel_ir.splitlines():
        if line.startswith("target triple"):
            triple = line.split('"')[1]
        elif line.startswith("target datalayout"):
            datalayout = line.split('"')[1]
    if triple != "nvptx64-nvidia-cuda" or not datalayout:
        raise CUDAEnzymeError("Numba-CUDA-MLIR emitted an unexpected LLVM target")
    return triple, datalayout


MODES = (
    "grad",
    "jvp",
    "jacfwd",
    "vjp",
    "jacrev",
)

# Forward Jacobians need a primal with several outputs, which means a
# tuple-returning one. Reverse products also accept a scalar return, and so
# does `jvp`: a scalar primal's directional derivative is a scalar, a
# tuple-returning one's is the whole tangent vector, and both are one sweep.
# `jvp` also composes with itself -- `jvp(jvp(f))` is the second-order
# directional derivative -- but only for a tuple-returning primal, because only
# the array call shape can carry the extra direction sets.
SCALAR_ONLY_MODES = frozenset({"grad"})
TUPLE_ONLY_MODES = frozenset({"jacfwd"})
TUPLE_FORWARD_MODES = frozenset({"jacfwd", "jvp"})
REVERSE_MODES = frozenset({"vjp", "jacrev"})
# What `modes=None` means: the original scalar-return defaults.
DEFAULT_MODES = frozenset({"grad", "jvp"})

# Leading array arguments of each tuple-primal call shape, before the primal's
# scalars. A tuple-returning primal needs no output array, so neither the
# primal values nor a shadow work buffer cross the call: forward modes take
# Enzyme's tangent struct directly, and reverse modes seed a scalarised primal
# whose return is the cotangent-weighted sum of the outputs.
# ``(leading, directions)``: leading array arguments before the primal's own,
# and how many further copies of the primal's argument list follow as direction
# vectors. A direction set mirrors the primal's arguments exactly -- an array
# where the primal takes a tuple, a scalar where it takes a scalar -- so a seed
# never has to be materialised in a shape the caller does not already have.
_TUPLE_CALL_LAYOUT = {
    "jacfwd": (1, 0),  # (jacobian, *args)
    "jvp": (1, 1),  # (tangent, *args, *directions); see _call_arities
    "vjp": (2, 0),  # (cotangent, gradient, *args)
    "jacrev": (1, 0),  # (jacobian, *args)
}


# How many direction sets a `jvp` call site may supply. The placeholder is
# generated wide enough for this many, and each count has its own arity, which
# is how a call site says how many sweeps it wants.
_MAX_DIRECTIONS = 8


def _call_arities(mode, n_params, depth=1, n_dirs=1):
    """
    Return how many arguments a call to `mode` takes for each primal shape.

    Parameters
    ----------
    mode : str
        Derivative mode.
    n_params : int
        Number of positional parameters of the primal's Python function.

    Returns
    -------
    dict
        Call arity keyed by ``"scalar"`` or ``"tuple"``, for each primal shape
        `mode` supports.

    See Also
    --------
    lazy_cuda_derivative : Dispatches on the arity of each call site.
    """
    arities = {}
    if mode not in TUPLE_ONLY_MODES:
        scalar = {"jvp": 2, "vjp": 2}
        arities["scalar"] = scalar.get(mode, n_params)
    if mode not in SCALAR_ONLY_MODES:
        leading, directions = _TUPLE_CALL_LAYOUT[mode]
        # Each composition level differentiates the level below with respect to
        # *all* of its arguments, so it takes a direction per argument and the
        # level's own argument list doubles: 2**(depth - 1) copies of the
        # primal's. A directional endpoint then adds a seed of that same width.
        groups = 2 ** (depth - 1)
        sets = n_dirs if directions else 0
        arities["tuple"] = leading + n_params * groups * (1 + sets)
    return arities


def _call_parameters(mode, n_params, depth=1):
    """
    Return the Python parameters shared by `mode`'s placeholder and bodies.

    Parameters
    ----------
    mode : str
        Derivative mode.
    n_params : int
        Number of positional parameters of the primal's Python function.

    Returns
    -------
    tuple of str
        ``a0, a1, ...`` up to the longest call shape, with every parameter
        beyond the shortest one defaulting to `None`.

    See Also
    --------
    _call_arities : The call shapes these parameters cover.
    """
    required = min(_call_arities(mode, n_params, depth, 1).values())
    widest = max(_call_arities(mode, n_params, depth, _MAX_DIRECTIONS).values())
    return tuple(
        f"a{index}" if index < required else f"a{index}=None" for index in range(widest)
    )


def _normalise_modes(modes) -> frozenset:
    """
    Validate a requested set of derivative entry points.

    Parameters
    ----------
    modes : iterable of str or None
        Subset of `MODES`. `None` means the original scalar-return defaults,
        `grad` and `jvp`; the Jacobian modes require an explicit request.

    Returns
    -------
    frozenset
        The requested modes.

    Raises
    ------
    CUDAEnzymeError
        If `modes` names an unknown mode or is empty.
    """

    if modes is None:
        return DEFAULT_MODES
    requested = frozenset(modes)
    unknown = requested - frozenset(MODES)
    if unknown:
        raise CUDAEnzymeError(
            f"unknown derivative mode(s) {sorted(unknown)}; expected {list(MODES)}"
        )
    if not requested:
        raise CUDAEnzymeError("at least one derivative mode must be requested")
    return requested


def _validate_modes_for_shape(requested, shape):
    """
    Reject derivative modes that do not match the primal ABI shape.

    Parameters
    ----------
    requested : frozenset
        Derivative modes to check.
    shape : str
        The primal's shape, as `_primal_shape` reports it.

    Raises
    ------
    CUDAEnzymeError
        If any requested mode is defined only for the other shape.

    See Also
    --------
    synthesise_cuda : Emits only the modes this accepts.
    """
    invalid = requested & (TUPLE_ONLY_MODES if shape == "scalar" else SCALAR_ONLY_MODES)
    if invalid:
        name = "scalar-return" if shape == "scalar" else "tuple-returning"
        raise CUDAEnzymeError(f"modes {sorted(invalid)} do not support a {name} primal")


# Stem of the external symbol names each reverse mode's per-partial entry
# points get, and whether those entry points take a trailing cotangent. A
# scalar primal's gradient and reverse Jacobian are the same sweep, differing
# only in the public wrapper built around them.
_SCALAR_REVERSE_ENTRIES = {
    "grad": ("grad", False),
    "vjp": ("vjp", True),
    "jacrev": ("jacrev", False),
}


def synthesise_cuda(
    kernel: CUDALoweredKernel,
    symbol_suffix: str,
    modes=None,
    depth=1,
    n_dirs=1,
    stage=None,
) -> CUDASynthesisedDriver:
    """
    Build C-ABI device wrappers containing Enzyme marker calls.

    Only the requested `modes` are emitted. Every entry point carries its own
    Enzyme marker call, and Enzyme differentiates each one, so emitting a mode
    that is never called is not free: the per-partial `grad` entry points alone
    cost one reverse differentiation each, and the whole set dominates build
    time for a many-argument primal.

    Parameters
    ----------
    kernel : CUDALoweredKernel
        Lowered primal device function.
    symbol_suffix : str
        Unique suffix for the generated external symbols.
    modes : iterable of str, optional
        Subset of `MODES` to emit. Defaults to `grad` and `jvp`.

    Returns
    -------
    CUDASynthesisedDriver
        Driver IR and the symbol names of the emitted entry points.

    See Also
    --------
    build_cuda : Runs Enzyme over the module this returns.
    """

    requested = _normalise_modes(modes)
    module = ir.Module(name="numba_enzyme_cuda_driver")
    module.triple, module.data_layout = _target_lines(kernel.ir)
    shape = _primal_shape(kernel.return_type)
    _validate_modes_for_shape(requested, shape)
    if shape == "tuple":
        return _synthesise_cuda_tuple(
            kernel, symbol_suffix, requested, module, depth, n_dirs, stage
        )

    if depth != 1:
        raise CUDAEnzymeError(
            "composing derivatives needs a tuple-returning primal; a "
            "scalar-return one has no array call shape to carry the extra "
            "direction sets"
        )
    scalar_type = _SCALAR_IR_TYPE[str(kernel.return_type)]
    n_args = len(kernel.arg_types)

    # Numba-CUDA-MLIR emits a device function compiled with abi="c" as a
    # direct scalar-returning function. The derivative entry points use the
    # same ABI and MLIR-compiled wrappers reconstruct the public tuple shapes.
    kernel_type = ir.FunctionType(scalar_type, [scalar_type] * n_args)
    kernel_fn = ir.Function(module, kernel_type, name=kernel.entry_symbol)

    i8p = ir.IntType(8).as_pointer()
    symbols = {}

    # Enzyme returns a bare scalar for one active by-value input and a literal
    # struct for multiple inputs.
    enzyme_grad_type = (
        scalar_type if n_args == 1 else ir.LiteralStructType([scalar_type] * n_args)
    )
    autodiff = None
    if requested & frozenset(_SCALAR_REVERSE_ENTRIES):
        autodiff = ir.Function(
            module,
            ir.FunctionType(enzyme_grad_type, [i8p], var_arg=True),
            name="__enzyme_autodiff",
        )

    def reverse_components(stem, weighted):
        """
        Emit the scalar C-ABI entry points composing one reverse-mode tuple.

        Each entry point is its own reverse sweep that keeps one partial
        derivative and discards the rest; a small MLIR device wrapper composes
        them into the public tuple result, avoiding the backend's unsupported
        external tuple-return ABI.

        Parameters
        ----------
        stem : str
            Middle portion of the generated external symbol names.
        weighted : bool
            Whether a trailing ``cotangent`` of the primal's scalar type
            scales the partial derivative. `False` emits the bare partials.

        Returns
        -------
        tuple of str
            The generated symbol names, one per primal argument.
        """
        extras = [scalar_type] if weighted else []
        generated = tuple(
            f"numba_enzyme_{stem}_{symbol_suffix}_{index}" for index in range(n_args)
        )
        for component, symbol in enumerate(generated):
            entry_fn = ir.Function(
                module,
                ir.FunctionType(scalar_type, [scalar_type] * n_args + extras),
                name=symbol,
            )
            for index, arg in enumerate(entry_fn.args[:n_args]):
                arg.name = f"x{index}"
            if weighted:
                entry_fn.args[n_args].name = "cotangent"

            builder = ir.IRBuilder(entry_fn.append_basic_block("entry"))
            gradient = builder.call(
                autodiff, [builder.bitcast(kernel_fn, i8p), *entry_fn.args[:n_args]]
            )
            value = (
                gradient if n_args == 1 else builder.extract_value(gradient, component)
            )
            if weighted:
                value = builder.fmul(value, entry_fn.args[n_args])
            builder.ret(value)
        return generated

    for mode, (stem, weighted) in _SCALAR_REVERSE_ENTRIES.items():
        if mode in requested:
            symbols[mode] = reverse_components(stem, weighted)

    if "jvp" in requested:
        # The MLIR wrapper flattens the two public tuples as x0..xn, dx0..dxn.
        # Enzyme receives each primal/tangent pair interleaved.
        enzyme_dup = ir.GlobalVariable(module, ir.IntType(32), name="enzyme_dup")
        enzyme_dup.linkage = "external"
        fwddiff = ir.Function(
            module,
            ir.FunctionType(scalar_type, [i8p], var_arg=True),
            name="__enzyme_fwddiff",
        )
        jvp_symbol = f"numba_enzyme_jvp_{symbol_suffix}"
        jvp_fn = ir.Function(
            module,
            ir.FunctionType(scalar_type, [scalar_type] * (2 * n_args)),
            name=jvp_symbol,
        )
        xs = list(jvp_fn.args[:n_args])
        dxs = list(jvp_fn.args[n_args:])
        for index, arg in enumerate(xs):
            arg.name = f"x{index}"
        for index, arg in enumerate(dxs):
            arg.name = f"dx{index}"

        builder = ir.IRBuilder(jvp_fn.append_basic_block("entry"))
        call_args = [builder.bitcast(kernel_fn, i8p)]
        for primal, tangent in zip(xs, dxs):
            call_args.extend([builder.load(enzyme_dup), primal, tangent])
        builder.ret(builder.call(fwddiff, call_args))
        symbols["jvp"] = (jvp_symbol,)

    return CUDASynthesisedDriver(ir=str(module), modes=requested, symbols=symbols)


_MEMREF_FIELDS = {
    1: ("allocated", "aligned", "offset", "size", "stride"),
    2: ("allocated", "aligned", "offset", "rows", "cols", "row_stride", "col_stride"),
}


def _declare_entry_point(module, name, params):
    """
    Declare one derivative entry point and name and split its arguments.

    Numba-CUDA-MLIR's C ABI expands each array parameter into
    ``{allocated, aligned, offset, sizes..., strides...}``, so an array arrives
    as that flat run of descriptor fields rather than as one pointer. A scalar
    parameter arrives as itself.

    Parameters
    ----------
    module : llvmlite.ir.Module
        Module to declare the entry point in.
    name : str
        External symbol name.
    params : sequence of tuple
        The entry point's parameters in order, each either
        ``("array", prefix, ndim)`` or ``("scalar", prefix, llvm_type)``.

    Returns
    -------
    entry_fn : llvmlite.ir.Function
        The declared entry point, with every argument named.
    handles : list
        One entry per element of `params`, in order: the tuple of descriptor
        fields for an array, or the value itself for a scalar.

    See Also
    --------
    _synthesise_cuda_tuple : Declares every entry point through this.
    """
    i8p = ir.IntType(8).as_pointer()
    i64 = ir.IntType(64)
    parameters = []
    for kind, _, detail in params:
        if kind == "array":
            parameters += [i8p, i8p] + [i64] * (len(_MEMREF_FIELDS[detail]) - 2)
        else:
            parameters.append(detail)

    entry_fn = ir.Function(
        module, ir.FunctionType(ir.VoidType(), parameters), name=name
    )
    handles = []
    position = 0
    for kind, prefix, detail in params:
        if kind == "array":
            fields = _MEMREF_FIELDS[detail]
            descriptor = tuple(entry_fn.args[position : position + len(fields)])
            for field, arg in zip(fields, descriptor):
                arg.name = f"{prefix}_{field}"
            handles.append(descriptor)
            position += len(fields)
        else:
            argument = entry_fn.args[position]
            argument.name = prefix
            handles.append(argument)
            position += 1
    return entry_fn, handles


def _synthesise_cuda_tuple(
    kernel: CUDALoweredKernel,
    symbol_suffix: str,
    requested,
    module,
    depth=1,
    n_dirs=1,
    stage=None,
) -> CUDASynthesisedDriver:
    """
    Build the derivative entry points for a tuple-returning primal.

    Numba-CUDA-MLIR lowers ``UniTuple(dtype, n)(x0, ..., xn)`` to an LLVM
    function returning a struct by value -- or the bare scalar, for one output.
    That needs no output array on either side of the call:

    - forward modes call ``__enzyme_fwddiff`` on the primal directly, and the
      marker returns the tangent struct, one sweep per column;
    - reverse modes differentiate an internal scalarisation
      ``g(x, w) = sum_k w_k * f_k(x)`` with the weights inactive, so one
      reverse sweep returns ``w @ J``. Enzyme does not accept an aggregate
      differential return on ``__enzyme_autodiff``, which rules out seeding
      the struct return itself.

    Parameters
    ----------
    kernel : CUDALoweredKernel
        Lowered tuple-returning primal.
    symbol_suffix : str
        Unique suffix for the generated external symbols.
    requested : frozenset
        Requested modes.
    module : llvmlite.ir.Module
        Module to emit into, already carrying the target lines.

    Returns
    -------
    CUDASynthesisedDriver
        Driver IR and the emitted symbol names, keyed by mode.
    """
    scalar_type = _SCALAR_IR_TYPE[str(kernel.return_type.dtype)]
    # Each primal argument is a scalar or a tuple; the entry points take a
    # tuple as an array and load its elements, so Enzyme sees the same flat
    # scalars however the caller spelled them.
    widths = _argument_widths(kernel.arg_types)
    n_args = _flat_argument_count(kernel.arg_types)
    n_out = kernel.return_type.count
    scalar_types = [scalar_type] * n_args
    i8p = ir.IntType(8).as_pointer()
    i32 = ir.IntType(32)
    i64 = ir.IntType(64)
    scalar_ptr = scalar_type.as_pointer()
    result_type = (
        scalar_type if n_out == 1 else ir.LiteralStructType([scalar_type] * n_out)
    )
    one = ir.Constant(scalar_type, 1.0)
    zero = ir.Constant(scalar_type, 0.0)

    primal_fn = ir.Function(
        module, ir.FunctionType(result_type, scalar_types), name=kernel.entry_symbol
    )
    enzyme_dup = ir.GlobalVariable(module, i32, name="enzyme_dup")
    enzyme_dup.linkage = "external"
    enzyme_const = ir.GlobalVariable(module, i32, name="enzyme_const")
    enzyme_const.linkage = "external"

    fwddiff = None
    # Inner composition levels are forward sweeps whatever the endpoint is, so
    # a reverse endpoint over a jvp still needs the forward marker declared.
    if requested & TUPLE_FORWARD_MODES or depth > 1:
        fwddiff = ir.Function(
            module,
            ir.FunctionType(result_type, [i8p], var_arg=True),
            name="__enzyme_fwddiff",
        )

    # Enzyme matches its markers by name *prefix*, so one declaration per
    # differentiated arity can coexist; a composition level changes that arity.
    scalarisations = {}
    reverse_markers = {}

    def component(builder, aggregate, count, index):
        """Return field `index` of an aggregate, or the value if it is scalar."""
        return aggregate if count == 1 else builder.extract_value(aggregate, index)

    def memref_base(builder, descriptor):
        """Return a pointer to the first logical element of a memref."""
        return builder.gep(
            builder.bitcast(descriptor[1], scalar_ptr), [descriptor[2]], inbounds=True
        )

    def forward(builder, callee, xs, column):
        """Emit one forward sweep seeding argument `column`; return the tangent."""
        dup = builder.load(enzyme_dup)
        call_args = [builder.bitcast(callee, i8p)]
        for index, primal in enumerate(xs):
            seed = builder.select(
                builder.icmp_signed("==", column, ir.Constant(i32, index)), one, zero
            )
            call_args.extend([dup, primal, seed])
        return builder.call(fwddiff, call_args)

    def seeded(builder, callee, xs, seeds):
        """Emit one forward sweep of `callee` with caller-supplied seeds.

        `forward` builds a unit seed per sweep, which is all a Jacobian column
        needs; a directional derivative wants an arbitrary seed. `callee` is
        the primal at depth one and the level below at any greater depth.
        """
        dup = builder.load(enzyme_dup)
        call_args = [builder.bitcast(callee, i8p)]
        for primal, seed in zip(xs, seeds):
            call_args.extend([dup, primal, seed])
        return builder.call(fwddiff, call_args)

    def scalarise(callee, count, outputs):
        """``g(x, w) = sum_k w_k * callee(x)_k``, the primal a reverse sweep seeds.

        Enzyme does not accept an aggregate differential return on
        ``__enzyme_autodiff``, so a multi-output function is differentiated
        through this weighted sum instead; one sweep then returns ``w @ J``.
        """
        existing = scalarisations.get(callee.name)
        if existing is not None:
            # Several reverse endpoints over the same level share one.
            return existing
        fn = ir.Function(
            module,
            ir.FunctionType(scalar_type, [scalar_type] * (count + outputs)),
            name=f"numba_enzyme_scalarised_{symbol_suffix}",
        )
        fn.linkage = "internal"
        body = ir.IRBuilder(fn.append_basic_block("entry"))
        values = body.call(callee, fn.args[:count])
        total = zero
        for index in range(outputs):
            field = values if outputs == 1 else body.extract_value(values, index)
            total = body.fadd(total, body.fmul(fn.args[count + index], field))
        body.ret(total)
        scalarisations[callee.name] = fn
        return fn

    def reverse(builder, callee, count, outputs, xs, weights):
        """Emit one reverse sweep of `callee`'s scalarisation; return the gradient."""
        marker = reverse_markers.get(count)
        if marker is None:
            marker = ir.Function(
                module,
                ir.FunctionType(
                    scalar_type
                    if count == 1
                    else ir.LiteralStructType([scalar_type] * count),
                    [i8p],
                    var_arg=True,
                ),
                name=f"__enzyme_autodiff_{count}",
            )
            reverse_markers[count] = marker
        const = builder.load(enzyme_const)
        call_args = [builder.bitcast(scalarise(callee, count, outputs), i8p), *xs]
        for weight in weights:
            call_args.extend([const, weight])
        return builder.call(marker, call_args)

    def unit_weights(builder, selected, count):
        """Build the one-hot cotangent picking output `selected` (``i64``)."""
        return [
            builder.select(
                builder.icmp_signed("==", selected, ir.Constant(i64, index)), one, zero
            )
            for index in range(count)
        ]

    def flatten(builder, handles):
        """Reconstitute the primal's flat scalars from an entry point's arguments.

        The loads happen before the Enzyme marker, so the primal it
        differentiates still takes scalars however the caller passed them.
        """
        xs = []
        for width, handle in zip(widths, handles):
            if width is None:
                xs.append(handle)
                continue
            base = memref_base(builder, handle)
            for element in range(width):
                offset = builder.mul(ir.Constant(i64, element), handle[4])
                xs.append(builder.load(builder.gep(base, [offset], inbounds=True)))
        return xs

    def store_vector(builder, aggregate, count, destination):
        """Store every field of an aggregate into a one-dimensional memref."""
        base = memref_base(builder, destination)
        for index in range(count):
            offset = builder.mul(ir.Constant(i64, index), destination[4])
            builder.store(
                component(builder, aggregate, count, index),
                builder.gep(base, [offset], inbounds=True),
            )

    def store_slice(builder, aggregate, count, matrix, index, along_rows):
        """Store an aggregate into one row or column of a two-dimensional memref.

        A forward sweep fills a Jacobian *column* -- it is seeded by an input --
        and a reverse sweep fills a *row*, seeded by an output; the two differ
        only in which stride the sweep index multiplies.
        """
        base = memref_base(builder, matrix)
        for position in range(count):
            offset = ir.Constant(i64, position)
            first, second = (index, offset) if along_rows else (offset, index)
            location = builder.add(
                builder.mul(first, matrix[5]), builder.mul(second, matrix[6])
            )
            builder.store(
                component(builder, aggregate, count, position),
                builder.gep(base, [location], inbounds=True),
            )

    def load_element(builder, matrix, row, column):
        """Load one element of a two-dimensional memref."""
        base = memref_base(builder, matrix)
        location = builder.add(
            builder.mul(row, matrix[5]),
            builder.mul(ir.Constant(i64, column), matrix[6]),
        )
        return builder.load(builder.gep(base, [location], inbounds=True))

    def sweep_loop(entry_fn, handles, bound, emit):
        """Emit a do-while loop over `bound`, calling `emit(builder, k, xs)`."""
        entry_block = entry_fn.append_basic_block("entry")
        loop_block = entry_fn.append_basic_block("loop")
        exit_block = entry_fn.append_basic_block("exit")
        builder = ir.IRBuilder(entry_block)
        # The argument loads are loop-invariant, so they belong here rather
        # than in the body that runs one sweep per row or column.
        xs = flatten_groups(builder, handles)
        builder.branch(loop_block)
        builder = ir.IRBuilder(loop_block)
        index = builder.phi(i64, name="k")
        index.add_incoming(ir.Constant(i64, 0), entry_block)
        emit(builder, index, xs)
        following = builder.add(index, ir.Constant(i64, 1))
        index.add_incoming(following, builder.block)
        builder.cbranch(
            builder.icmp_signed("<", following, bound), loop_block, exit_block
        )
        ir.IRBuilder(exit_block).ret_void()

    # The entry points mirror the primal's own argument list: a tuple argument
    # arrives as a contiguous array, a scalar argument as itself.
    def mirrored_params(prefix):
        """One parameter per primal parameter, of the same kind."""
        return [
            ("scalar", f"{prefix}{position}", scalar_type)
            if width is None
            else ("array", f"{prefix}{position}", 1)
            for position, width in enumerate(widths)
        ]

    argument_params = mirrored_params("x")
    n_parameters = len(widths)

    def flatten_groups(builder, handles):
        """Flatten any number of mirrored argument groups into flat scalars."""
        scalars = []
        for start in range(0, len(handles), n_parameters):
            scalars += flatten(builder, handles[start : start + n_parameters])
        return scalars

    # ---- inner composition levels ---------------------------------------
    # Every level below the endpoint is emitted as an internal *definition*, so
    # that one Enzyme pass resolves the whole nest of markers. Differentiating
    # an already-built derivative instead would present Enzyme with an external
    # declaration and no body, which is why composition happens here rather
    # than by re-entering the public API. A forward level differentiates the
    # level below with respect to all of its arguments, so it takes a direction
    # per argument -- the call widens by one copy of the primal's argument list
    # per level -- and keeps the output count.
    level_fn, level_args = primal_fn, n_args
    level_params = list(argument_params)
    for position in range(1, depth):
        inner = ir.Function(
            module,
            ir.FunctionType(result_type, [scalar_type] * (2 * level_args)),
            name=f"numba_enzyme_level{position}_{symbol_suffix}",
        )
        if stage == "endpoints":
            # An earlier Enzyme run already turned this level into a real
            # function; this stage links against it and only calls it.
            pass
        else:
            # It has to stay visible for the next stage's link to resolve it.
            inner.linkage = "external" if stage == "levels" else "internal"
            body = ir.IRBuilder(inner.append_basic_block("entry"))
            body.ret(
                seeded(
                    body,
                    level_fn,
                    list(inner.args[:level_args]),
                    list(inner.args[level_args:]),
                )
            )
        level_params += mirrored_params(f"d{position}_")
        level_fn, level_args = inner, 2 * level_args

    symbols = {}
    if stage == "levels":
        # This stage exists only to resolve the inner markers; the endpoint
        # goes in the next one.
        return CUDASynthesisedDriver(ir=str(module), modes=requested, symbols=symbols)

    if "jvp" in requested:
        # (tangent, <level arguments>, <direction>): one sweep for the whole
        # directional derivative, however many outputs the primal has. A
        # Jacobian column is the special case of a unit direction, so a caller
        # that wants J @ v pays one sweep rather than one per column.
        symbol = f"numba_enzyme_jvp_{symbol_suffix}"
        per_sweep = len(level_params) // n_parameters
        entry_fn, (tangent_out, *arguments) = _declare_entry_point(
            module,
            symbol,
            [
                ("array", "tangent", 2 if n_dirs > 1 else 1),
                *level_params,
                *[
                    parameter
                    for sweep in range(n_dirs)
                    for group in range(per_sweep)
                    for parameter in mirrored_params(f"s{sweep}_{group}_")
                ],
            ],
        )
        builder = ir.IRBuilder(entry_fn.append_basic_block("entry"))
        scalars = flatten_groups(builder, arguments)
        values = scalars[:level_args]
        for sweep in range(n_dirs):
            start = level_args * (1 + sweep)
            tangent = seeded(
                builder, level_fn, values, scalars[start : start + level_args]
            )
            if n_dirs > 1:
                store_slice(
                    builder,
                    tangent,
                    n_out,
                    tangent_out,
                    ir.Constant(i64, sweep),
                    along_rows=True,
                )
            else:
                store_vector(builder, tangent, n_out, tangent_out)
        builder.ret_void()
        symbols["jvp"] = (symbol,)

    if "jacfwd" in requested:
        # (jacobian, <level arguments>): one forward sweep per column.
        symbol = f"numba_enzyme_jacfwd_{symbol_suffix}"
        entry_fn, (jac, *arguments) = _declare_entry_point(
            module, symbol, [("array", "jac", 2), *level_params]
        )

        def fill_column(builder, column, xs):
            tangent = forward(builder, level_fn, xs, builder.trunc(column, i32))
            store_slice(builder, tangent, n_out, jac, column, along_rows=False)

        sweep_loop(entry_fn, arguments, jac[4], fill_column)
        symbols["jacfwd"] = (symbol,)

    if "vjp" in requested:
        # (cotangent, gradient, <level arguments>).
        symbol = f"numba_enzyme_vjp_{symbol_suffix}"
        rank = 2 if n_dirs > 1 else 1
        entry_fn, (cotangent, gradient, *arguments) = _declare_entry_point(
            module,
            symbol,
            [
                ("array", "cotangent", rank),
                ("array", "gradient", rank),
                *level_params,
            ],
        )
        if n_dirs > 1:
            # A row of cotangents per sweep. The count is the array's own
            # extent, so this is the same loop jacrev runs -- jacrev just
            # supplies the identity instead of reading the rows.
            def fill_row(builder, row, xs):
                weights = [
                    load_element(builder, cotangent, row, index)
                    for index in range(n_out)
                ]
                store_slice(
                    builder,
                    reverse(builder, level_fn, level_args, n_out, xs, weights),
                    level_args,
                    gradient,
                    row,
                    along_rows=True,
                )

            sweep_loop(entry_fn, arguments, cotangent[3], fill_row)
        else:
            builder = ir.IRBuilder(entry_fn.append_basic_block("entry"))
            xs = flatten_groups(builder, arguments)
            source = memref_base(builder, cotangent)
            weights = [
                builder.load(
                    builder.gep(
                        source,
                        [builder.mul(ir.Constant(i64, index), cotangent[4])],
                        inbounds=True,
                    )
                )
                for index in range(n_out)
            ]
            store_vector(
                builder,
                reverse(builder, level_fn, level_args, n_out, xs, weights),
                level_args,
                gradient,
            )
            builder.ret_void()
        symbols["vjp"] = (symbol,)

    if "jacrev" in requested:
        # (jacobian, <level arguments>): one reverse sweep per output row.
        symbol = f"numba_enzyme_jacrev_{symbol_suffix}"
        entry_fn, (jac, *arguments) = _declare_entry_point(
            module, symbol, [("array", "jac", 2), *level_params]
        )

        def fill_row(builder, row, xs):
            gradient = reverse(
                builder,
                level_fn,
                level_args,
                n_out,
                xs,
                unit_weights(builder, row, n_out),
            )
            store_slice(builder, gradient, level_args, jac, row, along_rows=True)

        sweep_loop(entry_fn, arguments, jac[3], fill_row)
        symbols["jacrev"] = (symbol,)

    return CUDASynthesisedDriver(ir=str(module), modes=requested, symbols=symbols)


def _package_version(name: str) -> str:
    """
    Return an installed package version for cache invalidation.

    Parameters
    ----------
    name : str
        Distribution name.

    Returns
    -------
    str
        Installed version, or ``"not-installed"``.
    """
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _source_text(func) -> str:
    """
    Produce a stable representation of a device function's implementation.

    Parameters
    ----------
    func : numba_cuda_mlir.descriptor.MLIRDispatcher
        Device dispatcher to inspect.

    Returns
    -------
    str
        Source text when available, otherwise a bytecode representation.
    """
    try:
        return inspect.getsource(func.py_func)
    except (OSError, TypeError):
        code = func.py_func.__code__
        return repr((code.co_code, code.co_consts, code.co_names))


def _cuda_cache_key(
    func, arg_types, return_type, cc, modes, kernel_ir, depth=1, n_dirs=1
) -> str:
    """
    Compute the cache key for a CUDA derivative specialization.

    Parameters
    ----------
    func : numba_cuda_mlir.descriptor.MLIRDispatcher
        Device function being differentiated.
    arg_types : tuple of numba.types.Type
        Concrete argument types.
    return_type : numba.types.Type
        Concrete return type.
    cc : tuple of int
        CUDA compute capability.
    modes : frozenset
        Derivative entry points the build will emit.
    kernel_ir : str
        Lowered IR of the primal. This is what actually gets differentiated,
        and it is the only part of the key that sees through a closure: two
        device functions can share source text, qualified name and signature
        while closing over different callees, and would otherwise collide.

    Returns
    -------
    str
        Hex-encoded SHA-256 cache key.
    """
    options = getattr(func, "targetoptions", {})
    material = {
        "modes": sorted(modes),
        "depth": depth,
        "directions": n_dirs,
        "kernel_ir": kernel_ir,
        "source": _source_text(func),
        "qualname": func.py_func.__qualname__,
        "args": [str(t) for t in arg_types],
        "return": str(return_type),
        "cc": list(cc),
        "fastmath": bool(options.get("fastmath", False)),
        "opt": options.get("opt"),
        "numba": nb.__version__,
        "llvmlite": llvmlite.__version__,
        "numba_cuda_mlir": _package_version("numba-cuda-mlir"),
        "cuda_bindings": _package_version("cuda-bindings"),
        "cuda_backend_schema": _CUDA_CACHE_SCHEMA,
        "toolchain": _toolchain_fingerprint(),
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def _device_signatures(arg_types, return_type, shape, depth=1, n_dirs=1) -> dict:
    """
    Construct public Numba signatures for the derivative device calls.

    Parameters
    ----------
    arg_types : tuple of numba.types.Type
        Primal argument types.
    return_type : numba.types.Type
        Primal return type.
    shape : str
        The primal's shape, as `_primal_shape` reports it.

    Returns
    -------
    dict
        Signature of each mode the primal's shape supports, keyed by mode.
        The shapes the other one supports are absent.

    See Also
    --------
    load_cuda : Declares the device calls these signatures describe.
    """
    cuda_types = _cuda_imports()["types"]
    if shape == "tuple":
        # No output array: forward modes return Enzyme's tangent struct, and
        # reverse modes seed a scalarised primal, so only results cross. The
        # call mirrors the primal's own argument list, a tuple argument
        # supplied as an array of its elements. That one is layout "A": the
        # entry point reads it through the memref's own stride, so it does not
        # need to be contiguous, and a caller passing a row of a larger array
        # should not have to prove that it is.
        array = cuda_types.Array(return_type.dtype, 1, "C")
        matrix = cuda_types.Array(return_type.dtype, 2, "C")
        supplied = cuda_types.Array(return_type.dtype, 1, "A")
        scalars = tuple(
            supplied if width is not None else argument
            for argument, width in zip(arg_types, _argument_widths(arg_types))
        )
        level = scalars * (2 ** (depth - 1))
        # One sweep writes a vector; several write a matrix, one slice each.
        sweeps = matrix if n_dirs > 1 else array
        return {
            "jacfwd": cuda_types.void(matrix, *level),
            "jvp": cuda_types.void(sweeps, *level, *(level * n_dirs)),
            "vjp": cuda_types.void(sweeps, sweeps, *level),
            "jacrev": cuda_types.void(matrix, *level),
        }
    tuple_type = cuda_types.UniTuple(return_type, len(arg_types))
    return {
        "grad": tuple_type(*arg_types),
        "jvp": return_type(tuple_type, tuple_type),
        "vjp": tuple_type(tuple_type, return_type),
        "jacrev": tuple_type(*arg_types),
    }


def _sanitize_for_libnvvm(llvm_ir: str) -> str:
    """
    Make Enzyme's LLVM-15 output acceptable to libNVVM.

    Parameters
    ----------
    llvm_ir : str
        Differentiated LLVM IR.

    Returns
    -------
    str
        IR with unsupported attributes removed and math calls mapped to
        libdevice symbols.
    """

    # LLVM70's diagnostic text omits the i128 alignment that libNVVM's text
    # reader requires, even though the in-process bitcode path accepts it.
    if "-i128:" not in llvm_ir:
        llvm_ir = llvm_ir.replace("-i64:64:64-f32", "-i64:64:64-i128:128:128-f32")

    # Enzyme emits attributes such as ``mustprogress`` that are newer than the
    # IR dialect accepted by CUDA 12's libNVVM.
    unsupported_attributes = (
        "mustprogress",
        "nocallback",
        "nofree",
        "nosync",
        "willreturn",
    )
    for attribute in unsupported_attributes:
        llvm_ir = re.sub(rf"\b{attribute}[ \t]*", "", llvm_ir)
    # Keep referenced attribute groups non-empty for libNVVM's LLVM 7 reader.
    llvm_ir = re.sub(
        r"^(attributes #\d+ = \{)\s*(\})$",
        r"\1 nounwind \2",
        llvm_ir,
        flags=re.MULTILINE,
    )

    # LLVM 8 added ``fneg``, and fast-math flags on ``select`` and ``phi``.
    # Enzyme emits all three freely -- ``fneg`` for any negated adjoint, a
    # flagged ``select`` wherever it guards a singular derivative -- and
    # libNVVM's LLVM 7 text reader stops at the opcode it does not know.
    # Rewrite ``fneg`` to the spelling LLVM itself used before it existed, and
    # drop the flags, which only ever licensed optimisations.
    llvm_ir = re.sub(
        r"\bfneg\b((?:[ \t]+(?:fast|nnan|ninf|nsz|arcp|contract|afn|reassoc))*)"
        r"[ \t]+(half|float|double|fp128)[ \t]+",
        r"fsub\1 \2 -0.000000e+00, ",
        llvm_ir,
    )
    llvm_ir = re.sub(
        r"(=[ \t]+)(select|phi)"
        r"(?:[ \t]+(?:fast|nnan|ninf|nsz|arcp|contract|afn|reassoc))+[ \t]+",
        r"\1\2 ",
        llvm_ir,
    )

    # Enzyme expresses derivatives of libdevice calls using standard LLVM math
    # intrinsics.  CUDA 12's libNVVM rejects several of those intrinsics, but
    # provides equivalent libdevice functions. Translate derivative calls to
    # that ABI before libdevice is linked by Numba-CUDA-MLIR.
    math_intrinsics = {
        "llvm.sin.f32": "__nv_sinf",
        "llvm.sin.f64": "__nv_sin",
        "llvm.cos.f32": "__nv_cosf",
        "llvm.cos.f64": "__nv_cos",
        "llvm.exp.f32": "__nv_expf",
        "llvm.exp.f64": "__nv_exp",
        "llvm.exp2.f32": "__nv_exp2f",
        "llvm.exp2.f64": "__nv_exp2",
        "llvm.log.f32": "__nv_logf",
        "llvm.log.f64": "__nv_log",
        "llvm.log2.f32": "__nv_log2f",
        "llvm.log2.f64": "__nv_log2",
        "llvm.log10.f32": "__nv_log10f",
        "llvm.log10.f64": "__nv_log10",
        "llvm.sqrt.f32": "__nv_sqrtf",
        "llvm.sqrt.f64": "__nv_sqrt",
        "llvm.pow.f32": "__nv_powf",
        "llvm.pow.f64": "__nv_pow",
        "llvm.powi.f32.i32": "__nv_powif",
        "llvm.powi.f64.i32": "__nv_powi",
    }
    for intrinsic, libdevice in math_intrinsics.items():
        llvm_ir = llvm_ir.replace(f"@{intrinsic}", f"@{libdevice}")

    # Renaming an intrinsic may duplicate a libdevice declaration already
    # present for the primal.  Identical repeated declarations are rejected by
    # LLVM, so retain only the first declaration of each symbol.
    declarations = set()
    lines = []
    for line in llvm_ir.splitlines():
        match = re.match(r"^declare\b.*@([^ (]+)\(", line)
        if match and match.group(1) in declarations:
            continue
        if match:
            declarations.add(match.group(1))
        lines.append(line)
    return "\n".join(lines)


def _internalise_primal(llvm_ir: str, entry_symbol: str) -> str:
    """
    Give the linked primal internal linkage.

    The primal keeps external linkage through ``llvm-link`` so the driver's
    declaration resolves against it. Afterwards nothing outside the module
    needs it, and leaving it public would export it from the derivative PTX --
    where two derivatives of same-shaped primals would then define the same
    mangled symbol and collide in whichever kernel links both.

    Parameters
    ----------
    llvm_ir : str
        Linked module text.
    entry_symbol : str
        Mangled name of the primal.

    Returns
    -------
    str
        The module with the primal's definition marked ``internal``.

    Raises
    ------
    CUDAEnzymeError
        If the primal's definition is not found exactly once.
    """

    pattern = re.compile(
        rf'^define\s+(?!.*\b(?:internal|private)\b)(?=[^\n]*@"?{re.escape(entry_symbol)}"?\s*\()',
        re.MULTILINE,
    )
    internalised, count = pattern.subn("define internal ", llvm_ir)
    if count != 1:
        raise CUDAEnzymeError(
            f"expected exactly one definition of primal {entry_symbol!r} to "
            f"internalise, found {count}"
        )
    return internalised


def _resolve_specialization(func, signature, cc, modes):
    """
    Normalise and validate one CUDA derivative request.

    Parameters
    ----------
    func : numba_cuda_mlir.descriptor.MLIRDispatcher
        Device function to differentiate.
    signature : numba_cuda_mlir.typing.Signature or None
        Concrete specialization, otherwise inferred from annotations.
    cc : tuple of int or str or None
        CUDA compute capability.
    modes : iterable of str or None
        Requested derivative entry points.

    Returns
    -------
    arg_types : tuple of numba.types.Type
        Concrete argument types.
    return_type : numba.types.Type
        Concrete return type.
    requested : frozenset
        Validated derivative modes.
    compute_capability : tuple of int
        Resolved ``(major, minor)`` pair.
    shape : str
        The primal's shape, as `_primal_shape` reports it.

    Raises
    ------
    CUDAEnzymeError
        If the signature or the requested modes do not fit the primal.

    See Also
    --------
    build_cuda : Compiles the specialization this describes.
    differentiate_cuda : Caches the result of that compilation in process.
    """
    backend = _cuda_imports()
    arg_types, return_type = _normalise_signature(func, signature)
    requested = _normalise_modes(modes)
    shape = _primal_shape(return_type)
    _validate_modes_for_shape(requested, shape)
    _validate_primal_signature(shape, arg_types, return_type)
    return (
        arg_types,
        return_type,
        requested,
        _compute_capability(cc, backend["tools"]),
        shape,
    )


def build_cuda(
    func, signature=None, cc=None, modes=None, depth=1, n_dirs=1
) -> CUDABuiltKernel:
    """
    Build or load differentiated PTX for one CUDA specialization.

    Parameters
    ----------
    func : numba_cuda_mlir.descriptor.MLIRDispatcher
        Device function to differentiate.
    signature : numba_cuda_mlir.typing.Signature, optional
        Concrete specialization, otherwise inferred from annotations.
    cc : tuple of int or str, optional
        CUDA compute capability.
    modes : iterable of str, optional
        Derivative entry points to emit; see `MODES`. Defaults to `grad` and
        `jvp`.
        Each one costs its own Enzyme differentiation, so build only what will
        be called.

    Returns
    -------
    CUDABuiltKernel
        Differentiated LTO IR, external symbols, signatures, and cache
        metadata.
    """

    backend = _cuda_imports()
    arg_types, return_type, requested, compute_capability, shape = (
        _resolve_specialization(func, signature, cc, modes)
    )
    # Lowering is needed for the cache key, so it happens even on a hit. It is
    # the cheaper half of a build and the only way to key on what is actually
    # differentiated rather than on the primal's source text.
    kernel = lower_cuda(func, signature=signature, cc=compute_capability)
    cache_key = _cuda_cache_key(
        func,
        arg_types,
        return_type,
        compute_capability,
        requested,
        kernel.ir,
        depth,
        n_dirs,
    )
    entry_dir = _cache_dir() / "cuda" / cache_key
    # LTO IR rather than PTX: numba-cuda-mlir compiles the calling kernel to
    # LTO IR too whenever a link item is LTO IR, which lets nvJitLink inline
    # the derivative into its caller instead of leaving an opaque call with a
    # parameter per primal argument.
    code_path = entry_dir / "derivative.ltoir"
    meta_path = entry_dir / "meta.json"
    signatures = _device_signatures(arg_types, return_type, shape, depth, n_dirs)
    n_args = len(arg_types)

    if code_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        return CUDABuiltKernel(
            code=code_path.read_bytes(),
            modes=requested,
            symbols={mode: tuple(names) for mode, names in meta["symbols"].items()},
            signatures=signatures,
            n_args=n_args,
            shape=shape,
            path=code_path,
            from_cache=True,
            depth=depth,
            n_dirs=n_dirs,
        )

    toolchain = get_toolchain()
    entry_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = entry_dir / "kernel.ll"
    kernel_path.write_text(kernel.ir)

    # Enzyme preprocesses a callee before resolving a marker nested inside it,
    # so a reverse endpoint over a forward composition level cannot be built in
    # one pass: the inner marker is still a marker when the outer sweep reaches
    # it. Splitting the module in two and running Enzyme on each in turn gives
    # the outer sweep a real function to differentiate. Forward markers nest
    # happily, so everything else stays a single pass.
    staged = bool(requested & REVERSE_MODES) and depth > 1
    stages = ("levels", "endpoints") if staged else (None,)
    source_path = kernel_path
    driver = None
    for index, stage in enumerate(stages):
        driver = synthesise_cuda(
            kernel,
            cache_key[:24],
            modes=requested,
            depth=depth,
            n_dirs=n_dirs,
            stage=stage,
        )
        driver_path = entry_dir / f"driver{index}.ll"
        combined_path = entry_dir / f"combined{index}.ll"
        enzyme_path = entry_dir / f"enzyme_out{index}.ll"
        driver_path.write_text(driver.ir)
        subprocess.run(
            [
                str(toolchain.llvm_link),
                "-opaque-pointers=0",
                str(source_path),
                str(driver_path),
                "-S",
                "-o",
                str(combined_path),
            ],
            check=True,
        )
        if stage != "levels":
            # Only once the last stage has linked: the primal has to stay
            # visible for an intermediate stage to resolve against it.
            combined_path.write_text(
                _internalise_primal(combined_path.read_text(), kernel.entry_symbol)
            )
        subprocess.run(
            [
                str(toolchain.opt),
                "-opaque-pointers=0",
                f"-load-pass-plugin={toolchain.enzyme_plugin}",
                # The cleanup passes remove now-dead Enzyme marker declarations
                # and activity globals before libNVVM sees the module.
                # libNVVM's LLVM 7 text reader rejects unnamed/numeric function
                # arguments emitted by LLVM 15. instnamer makes the final
                # textual IR round-trip through libNVVM without changing
                # semantics.
                "-passes=enzyme,adce,globaldce,instnamer",
                "-S",
                str(combined_path),
                "-o",
                str(enzyme_path),
            ],
            check=True,
        )
        source_path = enzyme_path

    enzyme_ir = _sanitize_for_libnvvm(source_path.read_text())
    if re.search(r"\bcall\b[^\n]*@__enzyme_(?:autodiff|fwddiff)", enzyme_ir):
        raise CUDAEnzymeError("Enzyme left unresolved differentiation marker calls")
    cc_text = f"{compute_capability[0]}{compute_capability[1]}"
    target_options = {"fastmath": kernel.fastmath}
    code = backend["compile_to_ltoir"](
        enzyme_ir.encode(),
        backend["LibDevice"](),
        backend["nvvm_options"](cc_text, target_options),
    )
    code_path.write_bytes(code)
    meta_path.write_text(
        json.dumps(
            {
                "symbols": {
                    mode: list(names) for mode, names in driver.symbols.items()
                },
                "modes": sorted(driver.modes),
                "compute_capability": list(compute_capability),
            }
        )
    )
    return CUDABuiltKernel(
        code=code,
        modes=requested,
        symbols=driver.symbols,
        signatures=signatures,
        n_args=n_args,
        shape=shape,
        path=code_path,
        from_cache=False,
        depth=depth,
        n_dirs=n_dirs,
    )


def load_cuda(built: CUDABuiltKernel) -> CUDADifferentiable:
    """
    Wrap differentiated device code in call-site implementations.

    Each implementation takes the parameters of `lazy_cuda_derivative`'s
    placeholder for its mode and calls the linked entry points directly. A
    tuple-returning primal's entry point already has the public call shape. A
    scalar one's public shapes are tuples, which the C ABI cannot carry, so its
    implementation rebuilds them from one per-partial entry point per argument.

    Parameters
    ----------
    built : CUDABuiltKernel
        Compiled derivative artifact.

    Returns
    -------
    CUDADifferentiable
        Implementations for the modes `built` was built with.

    See Also
    --------
    differentiate_cuda : Caches the result of this in process.
    """

    backend = _cuda_imports()
    # A path is a stable, hashable link item. Numba-CUDA-MLIR deduplicates it
    # when several modes are used by one caller.
    link_item = str(built.path)
    n_args = built.n_args

    def external(symbol, signature):
        """
        Declare one entry point as an external CUDA device call.

        Parameters
        ----------
        symbol : str
            External symbol name in the linked device code.
        signature : numba.core.typing.Signature
            The call's concrete signature.

        Returns
        -------
        object
            Numba-CUDA-MLIR external device-call descriptor.
        """
        return backend["cuda"].declare_device(
            symbol, signature, link=link_item, abi="c"
        )

    if built.shape == "scalar":
        arg_types = built.signatures["grad"].args
        return_type = built.signatures["grad"].return_type.dtype
        primal_items = [f"a0[{index}]" for index in range(n_args)]

    implementations = {}
    externals = {}
    for mode in built.modes:
        parameters = _call_parameters(mode, n_args, built.depth)
        names = [parameter.split("=")[0] for parameter in parameters]
        symbols = built.symbols[mode]
        if built.shape == "tuple":
            arity = _call_arities(mode, n_args, built.depth, built.n_dirs)[built.shape]
            entry = external(symbols[0], built.signatures[mode])
            externals[mode] = entry
            namespace = {"_entry": entry}
            body = f"_entry({', '.join(names[:arity])})"
        elif mode == "jvp":
            # Both public tuples flatten to the entry point's x0..xn, dx0..dxn.
            tangent_items = [f"a1[{index}]" for index in range(n_args)]
            namespace = {
                "_entry": external(symbols[0], return_type(*arg_types, *arg_types))
            }
            body = f"return _entry({', '.join(primal_items + tangent_items)})"
        else:
            forwarded, extras = {
                "grad": (names[:n_args], ()),
                "jacrev": (names[:n_args], ()),
                "vjp": ([*primal_items, "a1"], (return_type,)),
            }[mode]
            signature = return_type(*arg_types, *extras)
            namespace = {
                f"_component{index}": external(symbol, signature)
                for index, symbol in enumerate(symbols)
            }
            calls = ", ".join(
                f"_component{index}({', '.join(forwarded)})" for index in range(n_args)
            )
            body = f"return ({calls}{',' if n_args == 1 else ''})"
        # Generated with a fixed arity so the MLIR frontend sees ordinary calls
        # and tuple construction rather than *args or a comprehension.
        implementations[mode] = _fixed_arity_function(
            f"_{mode}", parameters, body, namespace, f"<numba-enzyme-{mode}>"
        )

    return CUDADifferentiable(
        implementations=implementations, n_args=n_args, externals=externals
    )


def _primal_signature_for_call(func, arg_types, explicit_signature=None):
    """
    Resolve the concrete primal signature for one derivative call site.

    Parameters
    ----------
    func : numba_cuda_mlir.descriptor.MLIRDispatcher
        Primal device dispatcher.
    arg_types : tuple
        Concrete call-site argument types.
    explicit_signature : numba_cuda_mlir.typing.Signature, optional
        User-supplied specialization constraint.

    Returns
    -------
    numba_cuda_mlir.typing.Signature
        Concrete primal signature including its inferred return type.

    Raises
    ------
    CUDAEnzymeError
        If an explicit signature does not match the call site.
    """

    from numba_cuda_mlir.numba_cuda.types.misc import unliteral

    arg_types = tuple(unliteral(t) for t in arg_types)
    if explicit_signature is not None:
        expected_args, return_type = _normalise_signature(func, explicit_signature)
        if arg_types != expected_args:
            raise CUDAEnzymeError(
                f"CUDA derivative call has argument types {arg_types}, but its "
                f"explicit signature requires {expected_args}"
            )
        return return_type(*expected_args)

    # Device dispatchers normally discover their return type while the caller
    # is typed. Compile that same call specialization and use its inferred
    # signature as Enzyme's concrete primal ABI.
    compile_result = func._compile_as_device_callee(arg_types)
    return compile_result.signature


def _fixed_arity_function(name, parameters, body, namespace, filename):
    """
    Create a compiler-facing function with an exact Python signature.

    Parameters
    ----------
    name : str
        Generated function name.
    parameters : tuple of str
        Positional parameter names.
    body : str
        Function-body statement.
    namespace : dict
        Globals available to the generated function.
    filename : str
        Synthetic filename used in diagnostics.

    Returns
    -------
    callable
        Generated Python function.
    """

    parameter_text = ", ".join(parameters)
    source = f"def {name}({parameter_text}):\n    {body}"
    exec(  # noqa: S102 - identifiers and body are generated internally
        compile(source, filename, "exec"), namespace
    )
    return namespace[name]


# Second public argument of the two product modes, whose call sites pass
# tuples rather than the primal's flattened scalars.
def _inline_unless_star_call(expr, caller_info, callee_info):
    """
    Decide whether to inline one resolved derivative call.

    CPython compiles a call with more than 30 positional arguments as a
    star-args call, which Numba-CUDA-MLIR's inliner rejects. Such a call
    compiles its implementation as a separate function instead, which measured
    no slower once nvJitLink has linked the LTO IR. Every narrower call is
    inlined, which is also the only way a tuple argument reaches an
    implementation.

    Parameters
    ----------
    expr : numba.core.ir.Expr
        The call expression being resolved.
    caller_info : object
        Inlining information about the caller, unused.
    callee_info : object
        Inlining information about the implementation, unused.

    Returns
    -------
    bool
        Whether to inline the implementation at this call.

    See Also
    --------
    lazy_cuda_derivative : Registers this as its overload's inlining policy.
    """
    return expr.vararg is None


def lazy_cuda_derivative(func, mode, signature=None, cc=None, depth=1):
    """
    Create a derivative callable specialized during MLIR call-site typing.

    Nothing is compiled until an MLIR-backed caller is typed against the
    returned placeholder. The concrete argument types at that call site then
    give the primal's signature -- the two primal shapes, a scalar return and
    outputs written through a leading array, take different numbers of
    arguments -- and the derivative is built and cached for it.

    Parameters
    ----------
    func : numba_cuda_mlir.descriptor.MLIRDispatcher
        Primal device dispatcher.
    mode : str
        Derivative mode exposed by the returned callable; one of `MODES`.
    signature : numba_cuda_mlir.typing.Signature, optional
        Specialization every call site must resolve to.
    cc : tuple of int or str, optional
        CUDA compute capability passed to derivative compilation.

    Returns
    -------
    callable
        Compiler overload placeholder. It is not callable from the host.

    Raises
    ------
    CUDAEnzymeError
        If the primal does not have a supported positional signature.
    ValueError
        If `mode` is not a derivative mode.

    See Also
    --------
    differentiate_cuda : Builds the derivative each resolved call site needs.
    """

    from numba_cuda_mlir import extending
    from numba_cuda_mlir.numba_cuda.types.misc import unliteral

    if mode not in MODES:
        raise ValueError(f"unsupported CUDA derivative mode {mode!r}")

    primal_parameters = tuple(inspect.signature(func.py_func).parameters.values())
    positional_kinds = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }
    if not primal_parameters or any(
        parameter.kind not in positional_kinds for parameter in primal_parameters
    ):
        raise CUDAEnzymeError(
            "lazy CUDA differentiation requires one or more positional parameters"
        )
    n_params = len(primal_parameters)
    arities = _call_arities(mode, n_params, depth)
    parameters = _call_parameters(mode, n_params, depth)
    names = tuple(parameter.split("=")[0] for parameter in parameters)
    primal_name = func.py_func.__name__

    def resolve(arg_types):
        """
        Build `mode` for one call site and return its implementation.

        Parameters
        ----------
        arg_types : tuple
            Types of every placeholder parameter at the call site. Parameters
            the call omitted are `numba.types.Omitted`.

        Returns
        -------
        callable
            Implementation compiled in place of the placeholder.

        Raises
        ------
        CUDAEnzymeError
            If the call matches neither call shape, or its arguments do not
            fit the primal they describe.
        """
        cuda_types = _cuda_imports()["types"]
        supplied = tuple(
            unliteral(t)
            for t in arg_types
            if t is not None and not isinstance(t, cuda_types.Omitted)
        )
        # A `jvp` call site says how many sweeps it wants by how many mirrored
        # direction sets it passes, so each count has its own arity; a `vjp`
        # says so by the rank of its cotangent, whose extent then gives the
        # count at run time.
        shape, sweeps = None, 1
        if arities.get("scalar") == len(supplied):
            shape = "scalar"
        else:
            for candidate in range(1, _MAX_DIRECTIONS + 1):
                if _call_arities(mode, n_params, depth, candidate).get("tuple") == len(
                    supplied
                ):
                    shape, sweeps = "tuple", candidate
                    break
        if shape == "tuple" and mode == "vjp":
            cotangent = supplied[0]
            if isinstance(cotangent, cuda_types.Array) and cotangent.ndim == 2:
                sweeps = 2
        if shape is None:
            expected = " or ".join(str(arity) for arity in arities.values())
            raise CUDAEnzymeError(
                f"CUDA {mode} of {primal_name} takes {expected} arguments, "
                f"not {len(supplied)}"
            )

        if shape == "tuple":
            leading = _TUPLE_CALL_LAYOUT[mode][0]
            primal_args = tuple(supplied[leading : leading + n_params])
            if signature is not None:
                # A tuple argument is passed as an array, whose type cannot say
                # how long the tuple is -- so the declared signature supplies
                # those positions, and the call only has to get array-ness right.
                declared, _ = _normalise_signature(func, signature)
                widths = _argument_widths(declared)
                for position, width in enumerate(widths):
                    if width is None or position >= len(primal_args):
                        continue
                    if not isinstance(primal_args[position], cuda_types.Array):
                        raise CUDAEnzymeError(
                            f"{primal_name} takes {declared[position]} as argument "
                            f"{position}, so CUDA {mode} wants a contiguous array "
                            f"there, not {primal_args[position]}"
                        )
                primal_args = tuple(
                    given if width is None else declared[position]
                    for position, (given, width) in enumerate(zip(primal_args, widths))
                )
        elif mode in ("jvp", "vjp"):
            if not isinstance(supplied[0], cuda_types.BaseTuple):
                raise CUDAEnzymeError(f"CUDA {mode} expects a primal tuple")
            primal_args = tuple(unliteral(t) for t in supplied[0].types)
        else:
            primal_args = supplied[:n_params]

        resolved = _primal_signature_for_call(
            func, primal_args, explicit_signature=signature
        )
        resolved_shape = _primal_shape(resolved.return_type)
        if resolved_shape != shape:
            # Naming the arity the primal's real shape wants is more use than
            # listing every shape's, now that both share a placeholder.
            wanted = arities.get(resolved_shape)
            mismatch = (
                f"so CUDA {mode} takes {wanted} arguments, not {len(supplied)}"
                if wanted is not None
                else f"which does not match this CUDA {mode} call"
            )
            raise CUDAEnzymeError(
                f"{primal_name} returns {resolved.return_type}, {mismatch}"
            )
        if shape == "scalar" and mode == "jvp":
            other = supplied[1]
            if not isinstance(other, cuda_types.BaseTuple):
                raise CUDAEnzymeError("CUDA jvp expects a tangent tuple")
            if tuple(unliteral(t) for t in other.types) != primal_args:
                raise CUDAEnzymeError(
                    "CUDA jvp primal and tangent tuples must have identical types"
                )
        elif shape == "scalar" and mode == "vjp":
            if supplied[1] != resolved.return_type:
                raise CUDAEnzymeError(
                    "CUDA vjp cotangent must match the primal's scalar return type"
                )

        differentiated = differentiate_cuda(
            func, signature=resolved, cc=cc, modes=(mode,), depth=depth, n_dirs=sweeps
        )
        return differentiated.implementations[mode]

    placeholder = _fixed_arity_function(
        f"_lazy_{mode}",
        parameters,
        "raise RuntimeError('CUDA derivatives are device callables')",
        {},
        f"<numba-enzyme-lazy-{mode}>",
    )
    resolver = _fixed_arity_function(
        f"_resolve_{mode}",
        parameters,
        f"return _resolve(({', '.join(names)},))",
        {"_resolve": resolve},
        f"<numba-enzyme-lazy-{mode}-resolver>",
    )
    extending.overload(
        placeholder,
        typing_registry=extending.typing_registry,
        inline=_inline_unless_star_call,
    )(resolver)
    placeholder.__name__ = f"{mode}_{primal_name}"
    placeholder.__qualname__ = placeholder.__name__
    placeholder._numba_enzyme_primal = func
    placeholder._numba_enzyme_mode = mode
    placeholder._numba_enzyme_depth = depth
    return placeholder


def differentiate_cuda(
    func, signature=None, cc=None, modes=None, depth=1, n_dirs=1
) -> CUDADifferentiable:
    """
    Compile and load CUDA derivative modes for a device function.

    Parameters
    ----------
    func : numba_cuda_mlir.descriptor.MLIRDispatcher
        Device function to differentiate.
    signature : numba_cuda_mlir.typing.Signature, optional
        Concrete specialization, otherwise inferred from annotations.
    cc : tuple of int or str, optional
        CUDA compute capability.
    modes : iterable of str, optional
        Derivative entry points to build; see `MODES`. Defaults to `grad` and
        `jvp`.

    Returns
    -------
    CUDADifferentiable
        Cached call-site implementations for `modes`.

    See Also
    --------
    lazy_cuda_derivative : Defers this until an MLIR call site is typed.
    """

    arg_types, return_type, requested, compute_capability, _ = _resolve_specialization(
        func, signature, cc, modes
    )
    # The on-disk cache keys on the lowered IR, which costs a lowering to
    # compute. In-process the dispatcher's own identity already distinguishes
    # everything that key would, without that cost.
    cache_key = (
        func,
        arg_types,
        return_type,
        compute_capability,
        requested,
        depth,
        n_dirs,
    )
    with _LOAD_LOCK:
        existing = _LOADED_CUDA_DERIVATIVES.get(cache_key)
        if existing is not None:
            return existing
        differentiated = load_cuda(
            build_cuda(
                func,
                signature=signature,
                cc=compute_capability,
                modes=requested,
                depth=depth,
                n_dirs=n_dirs,
            )
        )
        _LOADED_CUDA_DERIVATIVES[cache_key] = differentiated
        return differentiated
