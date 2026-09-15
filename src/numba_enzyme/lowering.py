"""
Lower a Python function to LLVM IR via Numba.

Compiles a Python function to LLVM IR with Numba, using the function's
own type annotations (:mod:`numba_enzyme.types` classes) to build the
Numba signature.The recovered kernel's exact parameter types are validated
against Numba's known ABI shape for use by driver synthesis.

See Also
--------
numba_enzyme.driver.synthesise : Consumes the `LoweredKernel`.

Examples
--------
>>> from numba_enzyme.lowering import lower
>>> from numba_enzyme.types import Float64
>>> def f(x: Float64) -> Float64:
...     return x * x
>>> lower(f).arg_types  # doctest: +SKIP
('double*', '{ i8*, i32, i8*, i8*, i32 }**', 'double')
"""

import inspect
import typing
from collections.abc import Callable
from dataclasses import dataclass

import llvmlite.binding as llvm_binding
import numba as nb

_CFUNC_PREFIX = "cfunc."

# Numba's fixed internal exception-info representation. Stable across
# signatures/arities in principle, but re-checked on every lowering
# for the sake of sanity.
_EXPECTED_EXCINFO_TYPE = "{ i8*, i32, i8*, i8*, i32 }**"

# LLVM textual type for each scalar numba type this package accepts as
# an annotation (see numba_enzyme.types). CPU vector outputs are represented
# by fixed-size homogeneous tuples of these scalar types.
# TODO: expand for other scalar types and arrays
_LLVM_SCALAR_TYPE = {
    nb.types.float64: "double",
    nb.types.float32: "float",
    nb.types.int32: "i32",
    nb.types.int64: "i64",
}

llvm_binding.initialize()


class LoweringError(RuntimeError):
    """
    Raise when Numba's emitted IR doesn't match the expected ABI.

    Covers a missing/unresolvable type annotation, an unsupported
    scalar type, and any deviation from the retptr/excinfo entry-point
    shape (wrong parameter count, name, or LLVM type).

    See Also
    --------
    lower : Raises this error when validation fails.

    Examples
    --------
    >>> from numba_enzyme.lowering import LoweringError, lower
    >>> def f(x):  # missing type annotations
    ...     return x
    >>> try:
    ...     lower(f)
    ... except LoweringError as exc:
    ...     print(exc)  # doctest: +SKIP
    """


@dataclass(frozen=True)
class LoweredKernel:
    """
    A Numba-compiled kernel, validated against the expected ABI.

    Attributes
    ----------
    ir : str
        The full LLVM IR text Numba emitted for the compiled function.
    entry_symbol : str
        The mangled name of the retptr/excinfo-ABI entry point.
    n_args : int
        Number of scalar arguments `entry_symbol` takes.
    n_outputs : int
        Number of scalar results. One denotes a scalar return; larger values
        denote a fixed-size homogeneous tuple return.
    return_type : str
        LLVM textual type of one result component.
    arg_types : tuple of str
        The entry point's real LLVM parameter types in order: the
        output pointer, the exception-info pointer, then `n_args`
        scalar types.

    See Also
    --------
    lower : Builds and validates a `LoweredKernel` instance.

    Examples
    --------
    >>> from numba_enzyme.lowering import lower
    >>> from numba_enzyme.types import Float64
    >>> def f(x: Float64) -> Float64:
    ...     return x * x
    >>> lower(f).n_args  # doctest: +SKIP
    1
    """

    ir: str
    entry_symbol: str
    n_args: int
    n_outputs: int
    return_type: str
    arg_types: tuple[str, ...]


def _numba_type_of(annotation) -> nb.types.Type:
    """
    Instantiate a `numba_enzyme.types` annotation to get its numba type.

    Parameters
    ----------
    annotation : type
        A `numba_enzyme.types` class, e.g. ``Float64``.

    Returns
    -------
    numba.core.types.Type
        The real numba type the annotation class's ``__new__`` returns.

    Raises
    ------
    LoweringError
        If `annotation` is not a callable `numba_enzyme.types`.

    Examples
    --------
    >>> from numba_enzyme.lowering import _numba_type_of
    >>> from numba_enzyme.types import Float64
    >>> _numba_type_of(Float64)
    float64
    """
    try:
        return annotation()
    except TypeError as exc:
        raise LoweringError(
            f"annotation {annotation!r} is not a numba_enzyme.types type"
        ) from exc


def _llvm_scalar_type(numba_type: nb.types.Type) -> str:
    """
    Map a numba scalar type to its LLVM textual form.

    Parameters
    ----------
    numba_type : numba.core.types.Type
        One of the scalar types `numba_enzyme.types` supports.

    Returns
    -------
    str
        The corresponding LLVM IR textual type, e.g. ``"double"``.

    Raises
    ------
    LoweringError
        If `numba_type` is not one of the supported scalar types.

    Examples
    --------
    >>> from numba_enzyme.lowering import _llvm_scalar_type
    >>> import numba as nb
    >>> _llvm_scalar_type(nb.types.float64)
    'double'
    """
    try:
        return _LLVM_SCALAR_TYPE[numba_type]
    except KeyError:
        raise LoweringError(
            f"unsupported scalar type {numba_type!r}; supported: "
            f"{sorted(str(t) for t in _LLVM_SCALAR_TYPE)}"
        ) from None


def _numba_return_type(annotation) -> tuple[nb.types.Type, nb.types.Type, int]:
    """
    Resolve a scalar or fixed homogeneous tuple return annotation.

    Parameters
    ----------
    annotation : object
        Evaluated Python return annotation.

    Returns
    -------
    return_type : numba.core.types.Type
        Complete Numba return type used to compile the function.
    scalar_type : numba.core.types.Type
        Scalar component type stored in the return value.
    n_outputs : int
        Number of scalar output components.

    Raises
    ------
    LoweringError
        If the annotation is an empty, variadic, or heterogeneous tuple.
    """
    if typing.get_origin(annotation) is not tuple:
        scalar_type = _numba_type_of(annotation)
        _llvm_scalar_type(scalar_type)
        return scalar_type, scalar_type, 1

    annotations = typing.get_args(annotation)
    if len(annotations) < 2:
        raise LoweringError("tuple return annotation must contain at least two types")
    if Ellipsis in annotations:
        raise LoweringError("tuple return annotation must have a fixed size")

    scalar_types = tuple(_numba_type_of(item) for item in annotations)
    for scalar_type in scalar_types:
        _llvm_scalar_type(scalar_type)
    if any(scalar_type != scalar_types[0] for scalar_type in scalar_types[1:]):
        raise LoweringError("tuple return annotation must be homogeneous")

    n_outputs = len(scalar_types)
    scalar_type = scalar_types[0]
    return nb.types.UniTuple(scalar_type, n_outputs), scalar_type, n_outputs


def lower(func: Callable) -> LoweredKernel:
    """
    Compile a Python function to a validated `LoweredKernel`.

    Reads `func`'s parameter and return type annotations to build the Numba
    signature. Parameters must use `numba_enzyme.types` classes; the return may
    also be a fixed-size homogeneous tuple of those classes. It compiles the
    function with :func:`numba.cfunc`, then locates and validates the resulting
    retptr/excinfo entry point.

    Parameters
    ----------
    func : callable
        A Python function whose scalar parameters use `numba_enzyme.types`
        annotations and whose return is a scalar or fixed homogeneous tuple of
        those types.

    Returns
    -------
    LoweredKernel
        The compiled, validated kernel.

    Raises
    ------
    LoweringError
        If `func` is missing a type annotation, uses an unsupported type, or
        Numba's emitted IR doesn't match the expected retptr/excinfo entry-point
        shape.

    See Also
    --------
    LoweredKernel : The validated result this function returns.

    Examples
    --------
    >>> from numba_enzyme.lowering import lower
    >>> from numba_enzyme.types import Float64
    >>> def f(x: Float64) -> Float64:
    ...     return x * x
    >>> lower(f).arg_types  # doctest: +SKIP
    ('double*', '{ i8*, i32, i8*, i8*, i32 }**', 'double')
    """
    hints = typing.get_type_hints(func)
    params = inspect.signature(func).parameters

    try:
        arg_numba_types = [_numba_type_of(hints[name]) for name in params]
        ret_numba_type, ret_scalar_type, n_outputs = _numba_return_type(hints["return"])
    except KeyError as exc:
        raise LoweringError(
            f"{func!r} is missing a type annotation for {exc.args[0]!r}"
        ) from exc

    sig = ret_numba_type(*arg_numba_types)
    compiled = nb.cfunc(sig, error_model="numpy")(func)
    ir_text = compiled.inspect_llvm()

    native_name = compiled.native_name
    if not native_name.startswith(_CFUNC_PREFIX):
        raise LoweringError(
            f"expected native_name to start with {_CFUNC_PREFIX!r}, got {native_name!r}"
        )
    # TODO: maybe better to use `replace(_CFUNC_PREFIX, "")`
    entry_symbol = native_name[len(_CFUNC_PREFIX) :]

    mod = llvm_binding.parse_assembly(ir_text)
    mod.verify()
    fn = next((f for f in mod.functions if f.name == entry_symbol), None)
    if fn is None:
        raise LoweringError(f"parsed IR has no function named {entry_symbol!r}")

    args = list(fn.arguments)
    n_args = len(arg_numba_types)
    expected_n_params = n_args + 2
    if len(args) != expected_n_params:
        raise LoweringError(
            f"expected {expected_n_params} parameters (retptr, excinfo, "
            f"{n_args} scalar args) but {entry_symbol!r} has {len(args)}"
        )

    retptr, excinfo, *scalar_args = args

    return_type = _llvm_scalar_type(ret_scalar_type)
    expected_retptr_type = (
        return_type + "*" if n_outputs == 1 else f"[{n_outputs} x {return_type}]*"
    )
    if retptr.name != "retptr" or str(retptr.type) != expected_retptr_type:
        raise LoweringError(
            f"expected first parameter 'retptr: {expected_retptr_type}', "
            f"got {retptr.name!r}: {retptr.type}"
        )
    if excinfo.name != "excinfo" or str(excinfo.type) != _EXPECTED_EXCINFO_TYPE:
        raise LoweringError(
            f"expected second parameter 'excinfo: {_EXPECTED_EXCINFO_TYPE}', "
            f"got {excinfo.name!r}: {excinfo.type}"
        )
    for arg, numba_type in zip(scalar_args, arg_numba_types):
        expected = _llvm_scalar_type(numba_type)
        if str(arg.type) != expected:
            raise LoweringError(
                f"expected parameter {arg.name!r} to be {expected}, got {arg.type}"
            )

    arg_types = tuple(str(a.type) for a in args)
    return LoweredKernel(
        ir=ir_text,
        entry_symbol=entry_symbol,
        n_args=n_args,
        n_outputs=n_outputs,
        return_type=return_type,
        arg_types=arg_types,
    )
