"""
Load a built shared object and expose it as plain Python callables.

Wraps a `numba_enzyme.build.BuiltKernel`'s ``jvp_<entry>``/``vjp_<entry>``
symbols with :mod:`ctypes` and derives every other mode from them: the
gradient and the reverse-mode Jacobians are vector-Jacobian products under
unit cotangents, and the forward-mode Jacobians are JVPs under unit tangents.
``vjp_<entry>`` is void and writes through one explicit output pointer per
scalar argument, uniformly for every arity, and accumulates an array argument's
gradient into a caller-owned shadow buffer (see `numba_enzyme.driver`).
For a scalar result, ``jvp_<entry>`` returns a bare scalar. For a tuple result,
it writes the output tangent through a trailing pointer.

`lazy_derivative` defers all of that for a function without annotations,
building one specialization for each tuple of argument types it is called with.

See Also
--------
numba_enzyme.build.build : Produces the `BuiltKernel` this module loads.
numba_enzyme.core.grad : Public API built on top of this module.

Examples
--------
>>> from numba_enzyme.build import build
>>> from numba_enzyme.runtime import load
>>> from numba_enzyme.types import Float64
>>> def f(x: Float64) -> Float64:
...     return x * x
>>> load(build(f)).grad(2.0)  # doctest: +SKIP
(4.0,)
"""

import ctypes
import inspect
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numba as nb
import numpy as np

from numba_enzyme.build import BuiltKernel, build
from numba_enzyme.lowering import ArgSpec

_CTYPES_SCALAR_TYPE = {
    "double": ctypes.c_double,
    "float": ctypes.c_float,
    "i32": ctypes.c_int32,
    "i64": ctypes.c_int64,
}

# Loaded derivatives of unannotated functions, keyed by function and argument
# types so every mode called with the same types shares one build.
_SPECIALIZATIONS = {}
_SPECIALIZATION_LOCK = threading.Lock()


def _array_ctypes_argtypes(spec: ArgSpec) -> list:
    """
    Provide ctypes argtypes for a given flattened array argument.

    Matches `numba_enzyme.driver` parameter group of an array:
    data pointer, data shadow pointer, nitems, itemsize,
    then `ndim` shape and `ndim` stride values.

    Parameters
    ----------
    spec : numba_enzyme.lowering.ArgSpec
        Must have `kind == "array"`.

    Returns
    -------
    list
        The ctypes types for this argument's flattened parameter group.
    """
    elem_ptr = ctypes.POINTER(_CTYPES_SCALAR_TYPE[spec.elem_llvm_type])
    return (
        [elem_ptr, elem_ptr, ctypes.c_int64, ctypes.c_int64]
        + [ctypes.c_int64] * spec.ndim
        + [ctypes.c_int64] * spec.ndim
    )


def _argument_ctypes(arg_specs, forward: bool) -> list:
    """
    Provide ctypes argtypes for the primal arguments of an entry point.

    Parameters
    ----------
    arg_specs : tuple of numba_enzyme.lowering.ArgSpec
        One specification per logical argument.
    forward : bool
        Whether the entry point is forward mode, which pairs every scalar
        argument with its tangent.

    Returns
    -------
    list
        The ctypes types for the primal arguments, in order.
    """
    argtypes = []
    for spec in arg_specs:
        if spec.kind == "scalar":
            ctype = _CTYPES_SCALAR_TYPE[spec.llvm_type]
            argtypes += [ctype, ctype] if forward else [ctype]
        else:
            argtypes += _array_ctypes_argtypes(spec)
    return argtypes


def _as_array_args(arr: np.ndarray, spec: ArgSpec):
    """
    Map a `numpy.ndarray` to the flattened array ABI.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to convert. Must be C-contiguous and have exactly
        `spec.ndim` dimensions. Currently only supports C-contiguous
        arrays. Non-contigous falls out of the scope and would certainly
        produce wrong results.
    spec : numba_enzyme.lowering.ArgSpec
        Must have `kind == "array"`.

    Returns
    -------
    Tuple[]
        Tuple containing five elements, namely data_ptr, nitems, itemsize,
        shape and strides. These are the values for a flattened array, except
        the shadow pointer, which the caller supplies separately.

    Raises
    ------
    ValueError
        If `arr`'s ndim doesn't match `spec.ndim` or it isn't
        C-contiguous.
    """
    if arr.ndim != spec.ndim:
        raise ValueError(f"expected an ndim={spec.ndim} array, got ndim={arr.ndim}")
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError("array arguments must be C-contiguous")
    elem_ctype = _CTYPES_SCALAR_TYPE[spec.elem_llvm_type]
    data_ptr = arr.ctypes.data_as(ctypes.POINTER(elem_ctype))
    return (
        data_ptr,
        arr.size,
        arr.itemsize,
        tuple(int(s) for s in arr.shape),
        tuple(int(s) for s in arr.strides),
    )


def _primal_args(arg_specs, xs, shadows) -> list:
    """
    Flatten primal values and their shadows into entry-point arguments.

    Parameters
    ----------
    arg_specs : tuple of numba_enzyme.lowering.ArgSpec
        One specification per logical argument.
    xs : tuple
        The primal values, a `float` per scalar argument and a
        `numpy.ndarray` per array argument.
    shadows : tuple
        One shadow per argument: in forward mode its tangent, in reverse mode
        `None` for a scalar and the zeroed gradient buffer for an array.

    Returns
    -------
    list
        The arguments to pass after any leading output pointers.

    Raises
    ------
    TypeError
        If a value's Python type doesn't match its declared scalar/array kind.
    ValueError
        If an array's shadow doesn't match its shape.
    """
    call_args = []
    for position, (spec, x, shadow) in enumerate(zip(arg_specs, xs, shadows)):
        if spec.kind == "scalar":
            if isinstance(x, np.ndarray):
                raise TypeError(
                    f"argument {position} expected a scalar, got an ndarray"
                )
            call_args.append(x)
            if shadow is not None:
                call_args.append(shadow)
            continue
        if not isinstance(x, np.ndarray) or not isinstance(shadow, np.ndarray):
            raise TypeError(
                f"argument {position} expected numpy.ndarray primal and tangent"
            )
        if shadow.shape != x.shape:
            raise ValueError(
                f"argument {position}'s tangent shape {shadow.shape} doesn't "
                f"match its primal shape {x.shape}"
            )
        data_ptr, nitems, itemsize, shape, strides = _as_array_args(x, spec)
        elem_ctype = _CTYPES_SCALAR_TYPE[spec.elem_llvm_type]
        d_data_ptr = shadow.ctypes.data_as(ctypes.POINTER(elem_ctype))
        call_args += [data_ptr, d_data_ptr, nitems, itemsize, *shape, *strides]
    return call_args


@dataclass(frozen=True)
class Differentiable:
    """
    Plain Python callables wrapping a built kernel's derivative symbols.

    Attributes
    ----------
    grad : callable
        Computes the reverse-mode gradient. Takes `n_args` positional
        arguments and returns a `tuple` of `n_args`; raises `TypeError`
        if called with the wrong number of arguments or on a vector result.
    jvp : callable
        Computes the forward-mode Jacobian-vector product. Takes a
        `tuple` of `n_args` primal values and a `tuple` of `n_args`
        tangent values, and returns a single `float`; raises
        `TypeError` if either tuple has the wrong length.
    jacfwd : callable
        Computes the full forward-mode Jacobian. Takes `n_args` positional
        arguments and returns a derivative tuple for a scalar result or an
        output-by-input tuple matrix for a vector result.
    vjp : callable
        Computes a reverse-mode vector-Jacobian product from primal values and
        an output cotangent.
    jacrev : callable
        Computes the full Jacobian in reverse mode, one output row per sweep.
    n_args : int
        Number of scalar arguments the underlying function takes.
    n_outputs : int
        Number of scalar output components.

    See Also
    --------
    load : Builds a `Differentiable` instance from a `BuiltKernel`.

    Examples
    --------
    >>> from numba_enzyme.build import build
    >>> from numba_enzyme.runtime import load
    >>> from numba_enzyme.types import Float64
    >>> def f(x: Float64) -> Float64:
    ...     return x * x
    >>> load(build(f)).n_args  # doctest: +SKIP
    1
    """

    grad: Callable[..., tuple[float, ...]]
    jvp: Callable[[tuple[float, ...], tuple[float, ...]], float | tuple[float, ...]]
    jacfwd: Callable[..., tuple]
    vjp: Callable[[tuple[float, ...], object], tuple[float, ...]]
    jacrev: Callable[..., tuple]
    n_args: int
    n_outputs: int


def _unit(length: int, index: int) -> tuple[float, ...]:
    """
    Build the one-hot seed selecting a single Jacobian row or column.

    Parameters
    ----------
    length : int
        Number of components in the seed.
    index : int
        Zero-based position of the single non-zero component.

    Returns
    -------
    tuple of float
        `length` values, all zero but the one at `index`.

    See Also
    --------
    load : Seeds the derivative callables it builds with this.

    Examples
    --------
    >>> from numba_enzyme.runtime import _unit
    >>> _unit(3, 1)
    (0.0, 1.0, 0.0)
    """
    return tuple(1.0 if position == index else 0.0 for position in range(length))


def load(built: BuiltKernel) -> Differentiable:
    """
    Load a built kernel via ctypes and wrap it as plain Python callables.

    Parameters
    ----------
    built : numba_enzyme.build.BuiltKernel
        The compiled shared object to load.

    Returns
    -------
    Differentiable
        Plain Python callables exposing every supported derivative mode.

    See Also
    --------
    Differentiable : The result this function returns.
    numba_enzyme.build.build : Produces the `built` this function
        consumes.

    Examples
    --------
    >>> from numba_enzyme.build import build
    >>> from numba_enzyme.runtime import load
    >>> from numba_enzyme.types import Float64
    >>> def f(x: Float64) -> Float64:
    ...     return x * x
    >>> load(build(f)).grad(2.0)  # doctest: +SKIP
    (4.0,)
    """
    lib = ctypes.CDLL(str(built.path))
    n = built.n_args
    m = built.n_outputs
    arg_specs = built.arg_specs
    scalar_ctypes = [
        _CTYPES_SCALAR_TYPE[spec.llvm_type]
        for spec in arg_specs
        if spec.kind == "scalar"
    ]
    return_ctype = _CTYPES_SCALAR_TYPE[built.return_type]
    has_array = any(spec.kind == "array" for spec in arg_specs)

    vjp_fn = getattr(lib, built.vjp_symbol)
    vjp_fn.restype = None
    vjp_fn.argtypes = [
        *[ctypes.POINTER(ctype) for ctype in scalar_ctypes],
        ctypes.POINTER(return_ctype),
        *_argument_ctypes(arg_specs, forward=False),
    ]

    jvp_fn = getattr(lib, built.jvp_symbol)
    jvp_fn.argtypes = _argument_ctypes(arg_specs, forward=True)
    if m == 1:
        jvp_fn.restype = return_ctype
    else:
        # A tuple result's tangent is written through a trailing out pointer.
        jvp_fn.restype = None
        jvp_fn.argtypes += (ctypes.POINTER(return_ctype),)

    def scalar_arguments_only(mode: str) -> None:
        """
        Reject a mode that does not support array arguments yet.

        Parameters
        ----------
        mode : str
            Name of the derivative mode, for the error message.

        Raises
        ------
        TypeError
            If the underlying function takes an array argument.
        """
        if has_array:
            raise TypeError(
                f"{mode} does not support array arguments yet; use grad or jvp"
            )

    def jvp(
        xs: tuple[float, ...], seed: tuple[float, ...]
    ) -> float | tuple[float, ...]:
        """
        Compute the forward-mode Jacobian-vector product.

        Parameters
        ----------
        xs : tuple
            The point to differentiate at, one value per argument (`float` or
            `numpy.ndarray`, matching each argument's declared kind).
        seed : tuple
            The tangent direction, one value per argument, matching `xs`'s
            shapes.

        Returns
        -------
        float or tuple of float
            The directional derivative of the underlying function at `xs` in
            direction `seed`. A vector-output function returns one tangent per
            output component.

        Raises
        ------
        TypeError
            If `xs` or `seed` doesn't have exactly `n` values.

        Examples
        --------
        >>> from numba_enzyme.build import build
        >>> from numba_enzyme.runtime import load
        >>> from numba_enzyme.types import Float64
        >>> def f(x: Float64) -> Float64:
        ...     return x * x
        >>> load(build(f)).jvp((2.0,), (1.0,))  # doctest: +SKIP
        4.0
        """
        if len(xs) != n or len(seed) != n:
            raise TypeError(f"expected {n} values for both xs and seed")
        call_args = _primal_args(arg_specs, xs, seed)
        if m == 1:
            return jvp_fn(*call_args)
        out = (return_ctype * m)()
        jvp_fn(*call_args, out)
        return tuple(out)

    def vjp(
        xs: tuple[float, ...], cotangent: float | tuple[float, ...]
    ) -> tuple[float, ...]:
        """
        Compute a reverse-mode vector-Jacobian product.

        Parameters
        ----------
        xs : tuple
            The point to differentiate at, one value per input argument
            (`float` or `numpy.ndarray`, matching each argument's declared
            kind).
        cotangent : float or tuple of float
            Scalar seed for a scalar result or one seed per vector component.

        Returns
        -------
        tuple
            Input cotangent, one value per primal argument: a `float` for a
            scalar argument, a `numpy.ndarray` of the same shape for an array
            argument.

        Raises
        ------
        TypeError
            If the primal or cotangent shape does not match the function.

        Examples
        --------
        >>> from numba_enzyme.build import build
        >>> from numba_enzyme.runtime import load
        >>> from numba_enzyme.types import Float64
        >>> def f(x: Float64) -> Float64:
        ...     return x * x
        >>> load(build(f)).vjp((2.0,), 1.0)  # doctest: +SKIP
        (4.0,)
        """
        if len(xs) != n:
            raise TypeError(f"expected {n} primal values, got {len(xs)}")
        if m == 1:
            if isinstance(cotangent, (tuple, list)):
                raise TypeError("expected a scalar cotangent for a scalar output")
            cotangents = (cotangent,)
        else:
            try:
                cotangents = tuple(cotangent)
            except TypeError:
                raise TypeError(f"expected {m} cotangent values") from None
            if len(cotangents) != m:
                raise TypeError(f"expected {m} cotangent values, got {len(cotangents)}")

        outputs = [ctype() for ctype in scalar_ctypes]
        gradients = [
            np.zeros_like(x)
            if spec.kind == "array" and isinstance(x, np.ndarray)
            else None
            for spec, x in zip(arg_specs, xs)
        ]
        vjp_fn(
            *[ctypes.byref(output) for output in outputs],
            (return_ctype * m)(*cotangents),
            *_primal_args(arg_specs, xs, gradients),
        )
        scalars = iter(output.value for output in outputs)
        return tuple(
            next(scalars) if spec.kind == "scalar" else gradient
            for spec, gradient in zip(arg_specs, gradients)
        )

    def grad(*xs: float) -> tuple[float, ...]:
        """
        Compute the reverse-mode gradient.

        Parameters
        ----------
        *xs : float or numpy.ndarray
            The point(s) to differentiate at, one per argument, matching each
            argument's declared scalar/array type.

        Returns
        -------
        tuple
            The gradient with respect to each argument: a `float` for a scalar
            argument, a `numpy.ndarray` of the same shape for an array
            argument.

        Raises
        ------
        TypeError
            If the underlying function has a vector result, or the number of
            arguments given doesn't match `n`.

        Examples
        --------
        >>> from numba_enzyme.build import build
        >>> from numba_enzyme.runtime import load
        >>> from numba_enzyme.types import Float64
        >>> def f(x: Float64) -> Float64:
        ...     return x * x
        >>> load(build(f)).grad(2.0)  # doctest: +SKIP
        (4.0,)
        """
        if m != 1:
            raise TypeError(
                "grad requires a scalar-output function; use jacfwd or jvp "
                "for tuple outputs"
            )
        if len(xs) != n:
            raise TypeError(f"expected {n} arguments, got {len(xs)}")
        return vjp(xs, 1.0)

    def jacfwd(*xs: float) -> tuple:
        """
        Compute the full forward-mode Jacobian.

        Parameters
        ----------
        *xs : float
            The point to differentiate at, one value per primal argument.

        Returns
        -------
        tuple
            For a scalar output, one partial derivative per primal argument.
            For a tuple output, an output-by-input tuple matrix.

        Raises
        ------
        TypeError
            If the number of arguments given doesn't match `n`.

        Examples
        --------
        >>> from numba_enzyme.build import build
        >>> from numba_enzyme.runtime import load
        >>> from numba_enzyme.types import Float64
        >>> def f(x: Float64) -> Float64:
        ...     return x * x
        >>> load(build(f)).jacfwd(2.0)  # doctest: +SKIP
        (4.0,)
        """
        scalar_arguments_only("jacfwd")
        if len(xs) != n:
            raise TypeError(f"expected {n} arguments, got {len(xs)}")
        columns = tuple(jvp(xs, _unit(n, column)) for column in range(n))
        if m == 1:
            return columns
        return tuple(
            tuple(columns[column][row] for column in range(n)) for row in range(m)
        )

    def jacrev(*xs: float) -> tuple:
        """
        Compute the full reverse-mode Jacobian.

        Parameters
        ----------
        *xs : float
            The point to differentiate at, one value per primal argument.

        Returns
        -------
        tuple
            A derivative tuple for a scalar result or an output-by-input tuple
            matrix for a vector result.

        Raises
        ------
        TypeError
            If the number of arguments given doesn't match `n`.

        Examples
        --------
        >>> from numba_enzyme.build import build
        >>> from numba_enzyme.runtime import load
        >>> from numba_enzyme.types import Float64
        >>> def f(x: Float64) -> Float64:
        ...     return x * x
        >>> load(build(f)).jacrev(2.0)  # doctest: +SKIP
        (4.0,)
        """
        scalar_arguments_only("jacrev")
        if len(xs) != n:
            raise TypeError(f"expected {n} arguments, got {len(xs)}")
        if m == 1:
            return vjp(xs, 1.0)
        return tuple(vjp(xs, _unit(m, row)) for row in range(m))

    return Differentiable(
        grad=grad,
        jvp=jvp,
        jacfwd=jacfwd,
        vjp=vjp,
        jacrev=jacrev,
        n_args=n,
        n_outputs=m,
    )


def _argument_type(value, position: int) -> nb.types.Type:
    """
    Infer the Numba type of one primal argument from its value.

    Parameters
    ----------
    value : object
        The argument as passed to a derivative callable.
    position : int
        Zero-based position of the argument, for error messages.

    Returns
    -------
    numba.core.types.Type
        Floating-point type of `value`.

    Raises
    ------
    TypeError
        If `value` is not a floating-point scalar. Integer arguments are
        rejected rather than silently promoted, since a derivative with
        respect to one is not defined.

    See Also
    --------
    lazy_derivative : Specializes on the types this infers.

    Examples
    --------
    >>> from numba_enzyme.runtime import _argument_type
    >>> _argument_type(1.0, 0)
    float64
    """
    if isinstance(value, np.ndarray):
        raise TypeError(
            f"argument {position} is an array, but inferring argument types does "
            "not support arrays yet; annotate the function with numba_enzyme.types"
        )
    try:
        numba_type = nb.typeof(value)
    except ValueError:
        numba_type = None
    if not isinstance(numba_type, nb.types.Float):
        raise TypeError(
            f"argument {position} is {type(value).__name__} {value!r}, but "
            "differentiation needs floating-point arguments"
        )
    return numba_type


def _specialization(func: Callable, arg_types: tuple) -> Differentiable:
    """
    Load, or reuse, the derivatives of a function for argument types.

    Parameters
    ----------
    func : callable
        Python function to differentiate.
    arg_types : tuple of numba.core.types.Type
        Concrete argument types to specialize for.

    Returns
    -------
    Differentiable
        Derivative callables for that specialization.

    See Also
    --------
    lazy_derivative : Resolves `arg_types` from each call.
    """
    key = (func, arg_types)
    with _SPECIALIZATION_LOCK:
        differentiated = _SPECIALIZATIONS.get(key)
        if differentiated is None:
            differentiated = load(build(func, arg_types))
            _SPECIALIZATIONS[key] = differentiated
    return differentiated


def lazy_derivative(func: Callable, mode: str) -> Callable:
    """
    Return a derivative callable that specializes on its argument types.

    Nothing is built until the callable is called. Each call infers the
    primal's argument types from its values, builds -- once per distinct tuple
    of types -- the same shared object `build` would for an annotated
    function, and forwards to its `mode` callable.

    Parameters
    ----------
    func : callable
        Python function to differentiate, with or without annotations.
    mode : str
        Name of a `Differentiable` derivative callable, such as ``"grad"``.

    Returns
    -------
    callable
        Host callable with the same call shape as ``load(built).<mode>``.

    See Also
    --------
    load : Builds the callables this forwards to.

    Examples
    --------
    >>> from numba_enzyme.runtime import lazy_derivative
    >>> def f(x, y):
    ...     return x * y
    >>> lazy_derivative(f, "grad")(2.0, 3.0)  # doctest: +SKIP
    (3.0, 2.0)
    """
    n = len(inspect.signature(func).parameters)
    by_python_types = {}

    def derivative(*args):
        """
        Compute the derivative, specializing on the primal's argument types.

        Parameters
        ----------
        *args : object
            Arguments in the call shape of ``load(built).<mode>``.

        Returns
        -------
        object
            That callable's result.

        Raises
        ------
        TypeError
            If the call passes the wrong number of primal values, or a primal
            value is not a floating-point scalar.
        """
        if mode in ("jvp", "vjp"):
            if len(args) != 2:
                raise TypeError(
                    f"{mode} expects a primal tuple and one more argument, "
                    f"got {len(args)} arguments"
                )
            primal = tuple(args[0])
        else:
            primal = args
        if len(primal) != n:
            raise TypeError(f"expected {n} primal values, got {len(primal)}")

        key = tuple(map(type, primal))
        differentiated = by_python_types.get(key)
        if differentiated is None:
            arg_types = tuple(
                _argument_type(value, position) for position, value in enumerate(primal)
            )
            differentiated = by_python_types[key] = _specialization(func, arg_types)
        return getattr(differentiated, mode)(*args)

    derivative.__name__ = f"{mode}_{func.__name__}"
    derivative.__qualname__ = derivative.__name__
    return derivative
