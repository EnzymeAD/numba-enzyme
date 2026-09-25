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
import re
import typing
from collections.abc import Callable
from dataclasses import dataclass

import llvmlite.binding as llvm_binding
import numba as nb

_CFUNC_PREFIX = "cfunc."

# Numba's execption data representation.
_EXPECTED_EXCINFO_TYPE = "{ i8*, i32, i8*, i8*, i32 }**"

# LLVM textual (bare) type for each numba scalar type.
# TODO: expand for other scalar types
_LLVM_SCALAR_TYPE = {
    nb.types.float64: "double",
    nb.types.float32: "float",
    nb.types.int32: "i32",
    nb.types.int64: "i64",
}

# Flattened array ABI of a Numba array on the LLVM IR level.
# struct fields in order: meminfo (i8*), parent (i8*), nitems (i64),
# itemsize (i64), data (<elem>*), shape[0..ndim-1] (i64 each),
# strides[0..ndim-1] (i64 each).
# TODO: not universal for every arch. Revisit this later
_ARRAY_HEADER_TYPES = ("i8*", "i8*", "i64", "i64")

# Matches a call to any Numba NRT (memory-management runtime) function,
# e.g. NRT_incref, NRT_MemInfo_alloc.
_NRT_CALL_RE = re.compile(r"\bcall\b[^\n]*@(NRT_\w+)")

# NRT_incref/NRT_decref are emitted by Numba as linkonce_odr functions
# with real bodies directly in the kernel IR and are not opaque. Enzyme's
# activity analysis can see straight through them.

# NRT_MemInfo_call_dtor is reachable transitively from NRT_decref
# path although the kernel IR never calls it directly. It is bodyless
# and its definition lives in `numba` `nrt.cpp` and it is linked
# during runtime.

# Allocation calls such as NRT_MemInfo_alloc_aligned and NRT_Free, etc,
# require enzyme_allocator/enzyme_deallocator/enzyme_type marker for array-output
# support. Not implemented yet.
_NRT_SUPPORTED_CALLS = frozenset({"NRT_incref", "NRT_decref", "NRT_MemInfo_call_dtor"})

# NRT_MemInfo_call_dtor is always declared as a bodyless function by Numba.
_NRT_DTOR_DECLARE_RE = re.compile(
    r"^(declare\s+void\s+@NRT_MemInfo_call_dtor\([^)]*\))(.*)$", re.MULTILINE
)

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
class ArgSpec:
    """
    Describes the ABI of one flattened logical argument.

    A logical argument is a single LLVM parameter associated with
    a scalar argument or an amalgamation of flattened LLVM parameters
    for an arrays. A scalar argument is identified by one LLVM parameter
    and an array in numba is represented by 5 + 2*ndim flattened LLVM
    parameters. For arrays we have `meminfo`, `parent`, `ntimes`,
    `itemsize`, `data` (5 in total) plus `shape` and `stride` for each
    `ndim` (hence 2 * ndim).

    Attributes
    ----------
    kind : str
        `scalar` or `array`, specifies the kind of the argument.
    llvm_type : str | None
        The LLVM type, e.g. `"double"`. This attribute gets assigned
        when `kind="scalar"` only, otherwise it is set to `None`.
    ndim : int | None
        The array's rank. This attribute gets assigned when `kind="array"`
        only, otherwise it is set to `None`.
    elem_llvm_type : str | None
        The LLVM type of the array's elements, e.g. `"double"`. This
        attribute gets assigned when `kind="array"` only, otherwise
        it is set to `None`.

    See Also
    --------
    LoweredKernel : Carries one `ArgSpec` per logical argument.

    Examples
    --------
    >>> from numba_enzyme.lowering import ArgSpec
    >>> ArgSpec(kind="array", ndim=1, elem_llvm_type="double").n_fields
    7
    """

    kind: str
    llvm_type: str | None = None
    ndim: int | None = None
    elem_llvm_type: str | None = None

    @property
    def n_fields(self) -> int:
        """
        Number of flattened LLVM parameters a given argument occupies.

        Returns
        -------
        int
            `1` for a scalar, `5 + 2*ndim` for an array.
        """
        return 1 if self.kind == "scalar" else 5 + 2 * self.ndim


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
        Number of logical arguments `entry_symbol` takes. Not to be
        confused with the number of flattened LLVM parameters, which
        may be larger once array arguments are involved.
    arg_types : tuple[str, ...]
        The entry point's flattened LLVM parameter types in the
        following order: the output pointer, the exception-info pointer,
        then one entry per flattened parameter (one per scalar argument,
        or `5 + 2*ndim` per array argument).
    arg_specs : tuple[ArgSpec, ...]
        One `ArgSpec` per logical argument, in declaration order,
        describing how each argument's fields map into `arg_types`.

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
    arg_types: tuple[str, ...]
    arg_specs: tuple[ArgSpec, ...]


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
        The numba type the annotation returns.

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
        The corresponding LLVM IR textual type, e.g. `"double"`.

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


def _validate_nrt_calls(ir_text: str) -> None:
    """
    Reject any NRT calls that are not whitelisted.

    Parameters
    ----------
    ir_text : str
        The kernel's LLVM IR text.

    Raises
    ------
    LoweringError
        If the kernel body calls an `NRT_*` function outside
        `_NRT_SUPPORTED_CALLS`.
    """
    for match in _NRT_CALL_RE.finditer(ir_text):
        name = match.group(1)
        if name in _NRT_SUPPORTED_CALLS:
            continue
        hint = (
            " memory allocation and array-output is not supported yet."
            if "Alloc" in name or name == "NRT_Free"
            else ""
        )
        raise LoweringError(
            f"unsupported kernel body: calls {name!r}{hint}. Only NRT "
            f"refcounting calls ({', '.join(sorted(_NRT_SUPPORTED_CALLS))}) "
            "are supported at the moment."
        )


# TODO: replace with a proper `ActivityAnalysis.cpp` NRT function
# table entry, similar to BLAS name table in Enzyme itself. That would
# let `NRT_incref`/`NRT_decref` be treated as known-inactive, instead of
# relying on their bodies being analysed by Enzyme.
def _annotate_nrt_dtor(ir_text: str) -> str:
    """
    Mark `NRT_MemInfo_call_dtor`'s declaration `nofree` and inactive.

    `NRT_MemInfo_call_dtor` is bodyless IR. It's definition is in `nrt.cpp`
    and is not linked into the kernel. It is reachable transitively via
    `NRT_decref` when refcount hits zero whenever a differentiated call decrefs
    an input array. Note that this call is required for the primal, but
    doesn't contribute to the derivative

    Parameters
    ----------
    ir_text : str
        The kernel's LLVM IR text.

    Returns
    -------
    str
        `ir_text`, with `nofree` and `"enzyme_inactive"` added to
        `NRT_MemInfo_call_dtor`'s declaration if present and not
        already marked.

    Notes
    -----
    `EnzymeLogic::CreateNoFree` and `ActivityAnalysis.cpp`.
    """

    def _add_attrs(match: re.Match) -> str:
        """
        Add `nonfree` and `enzyme_inactive` attributes.

        Parameters
        ----------
        match : re.Match
            Regex match.

        Returns
        -------
        str
            Updated NRT `dtor` declaration with `nofree` and
            `enzyme_inactive` attributes.
        """
        declare_head, rest = match.group(1), match.group(2)
        missing = [attr for attr in ("nofree", '"enzyme_inactive"') if attr not in rest]
        if not missing:
            return match.group(0)
        # New attributes must come after any linkage/visibility keyword
        # like `local_unnamed_addr`.
        return f"{declare_head}{rest} {' '.join(missing)}"

    return _NRT_DTOR_DECLARE_RE.sub(_add_attrs, ir_text)


def lower(func: Callable) -> LoweredKernel:
    """
    Compile a Python function to a validated `LoweredKernel`.

    Reads `func`'s parameter and return type annotations to build
    the Numba signature, compiles it with :func:`numba.cfunc`, then
    locates and validates the resulting retptr/excinfo entry point.

    Parameters
    ----------
    func : callable
        An annotated Python function whose.

    Returns
    -------
    LoweredKernel
        The compiled, validated kernel.

    Raises
    ------
    LoweringError
        If `func` is missing a type annotation, uses an unsupported
        type, or Numba's emitted IR doesn't match the expected
        retptr/excinfo entry-point shape.

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
        ret_numba_type = _numba_type_of(hints["return"])
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
    expected_n_flat_params = sum(
        5 + 2 * t.ndim if isinstance(t, nb.types.Array) else 1 for t in arg_numba_types
    )
    expected_n_params = expected_n_flat_params + 2
    if len(args) != expected_n_params:
        raise LoweringError(
            f"expected {expected_n_params} parameters (retptr, excinfo, "
            f"{expected_n_flat_params} flattened args) but {entry_symbol!r} "
            f"has {len(args)}"
        )

    retptr, excinfo, *flat_args = args

    expected_retptr_type = _llvm_scalar_type(ret_numba_type) + "*"
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

    arg_specs = []
    flat_idx = 0
    for numba_type in arg_numba_types:
        if isinstance(numba_type, nb.types.Array):
            ndim = numba_type.ndim
            elem_llvm_type = _llvm_scalar_type(numba_type.dtype)
            expected_fields = (
                *_ARRAY_HEADER_TYPES,
                elem_llvm_type + "*",
                *("i64",) * ndim,  # this is valid only for 64bit?
                *("i64",) * ndim,  # this is valid only for 64bit?
            )
            for expected in expected_fields:
                arg = flat_args[flat_idx]
                if str(arg.type) != expected:
                    raise LoweringError(
                        f"expected parameter {arg.name!r} to be {expected}, "
                        f"got {arg.type}"
                    )
                flat_idx += 1
            arg_specs.append(
                ArgSpec(kind="array", ndim=ndim, elem_llvm_type=elem_llvm_type)
            )
        else:
            expected = _llvm_scalar_type(numba_type)
            arg = flat_args[flat_idx]
            if str(arg.type) != expected:
                raise LoweringError(
                    f"expected parameter {arg.name!r} to be {expected}, got {arg.type}"
                )
            flat_idx += 1
            arg_specs.append(ArgSpec(kind="scalar", llvm_type=expected))

    # Only check NRT calls in the entry function.
    # The rest of the IR will be stripped out via
    # `llvm-extract`.
    _validate_nrt_calls(str(fn))
    ir_text = _annotate_nrt_dtor(ir_text)
    try:
        llvm_binding.parse_assembly(ir_text).verify()
    except RuntimeError as exc:
        raise LoweringError(
            f"internal error: NRT annotation produced invalid IR: {exc}"
        ) from exc

    arg_types = tuple(str(a.type) for a in args)
    return LoweredKernel(
        ir=ir_text,
        entry_symbol=entry_symbol,
        n_args=n_args,
        arg_types=arg_types,
        arg_specs=tuple(arg_specs),
    )
