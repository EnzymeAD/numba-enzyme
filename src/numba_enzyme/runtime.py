"""
Load a built shared object and expose it as plain Python callables.

Wraps a `numba_enzyme.build.BuiltKernel`'s `grad_<entry>``/
`jvp_<entry>` symbols with :mod:`ctypes`.
`grad_<entry>` is void and writes through an explicit output pointer
uniformly for every arity. `jvp_<entry>` always returns a bare `float`
regardless of arity.

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
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from numba_enzyme.build import BuiltKernel
from numba_enzyme.lowering import ArgSpec

# Maps an ArgSpec.elem_llvm_type string to its ctypes equivalent, for
# converting an array argument's data/shadow pointers.
# TODO: expand alongside numba_enzyme.lowering._LLVM_SCALAR_TYPE
_ELEM_CTYPE = {
    "double": ctypes.c_double,
    "float": ctypes.c_float,
    "i32": ctypes.c_int32,
    "i64": ctypes.c_int64,
}


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
    elem_ptr = ctypes.POINTER(_ELEM_CTYPE[spec.elem_llvm_type])
    return (
        [elem_ptr, elem_ptr, ctypes.c_int64, ctypes.c_int64]
        + [ctypes.c_int64] * spec.ndim
        + [ctypes.c_int64] * spec.ndim
    )


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
    elem_ctype = _ELEM_CTYPE[spec.elem_llvm_type]
    data_ptr = arr.ctypes.data_as(ctypes.POINTER(elem_ctype))
    return (
        data_ptr,
        arr.size,
        arr.itemsize,
        tuple(int(s) for s in arr.shape),
        tuple(int(s) for s in arr.strides),
    )


@dataclass(frozen=True)
class Differentiable:
    """
    Plain Python callables wrapping a built kernel's grad/JVP symbols.

    Attributes
    ----------
    grad : callable
        Computes the reverse-mode gradient. Takes `n_args` positional
        arguments and returns a `tuple` of `n_args`; raises `TypeError`
        if called with the wrong number of arguments.
    jvp : callable
        Computes the forward-mode Jacobian-vector product. Takes a
        `tuple` of `n_args` primal values and a `tuple` of `n_args`
        tangent values, and returns a single `float`; raises
        `TypeError` if either tuple has the wrong length.
    n_args : int
        Number of scalar arguments the underlying function takes.

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
    jvp: Callable[[tuple[float, ...], tuple[float, ...]], float]
    n_args: int


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
        Plain Python callables wrapping `built`'s grad/JVP symbols.

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
    arg_specs = built.arg_specs
    n_scalar = sum(1 for spec in arg_specs if spec.kind == "scalar")

    grad_argtypes = [ctypes.POINTER(ctypes.c_double)] if n_scalar else []
    jvp_argtypes = []
    for spec in arg_specs:
        if spec.kind == "scalar":
            grad_argtypes.append(ctypes.c_double)
            jvp_argtypes += [ctypes.c_double, ctypes.c_double]
        else:
            grad_argtypes += _array_ctypes_argtypes(spec)
            jvp_argtypes += _array_ctypes_argtypes(spec)

    grad_fn = getattr(lib, built.grad_symbol)
    grad_fn.restype = None
    grad_fn.argtypes = grad_argtypes

    jvp_fn = getattr(lib, built.jvp_symbol)
    jvp_fn.restype = ctypes.c_double
    jvp_fn.argtypes = jvp_argtypes

    def grad(*xs) -> tuple:
        """
        Compute the reverse-mode gradient.

        Parameters
        ----------
        *xs : float or numpy.ndarray
            The point(s) to differentiate at, one per argument,
            matching each argument's declared scalar/array type.

        Returns
        -------
        tuple
            The gradient with respect to each argument: a `float` for
            a scalar argument, a `numpy.ndarray` of the same shape for
            an array argument.

        Raises
        ------
        TypeError
            If the number of arguments given doesn't match `n_args`,
            or an argument's Python type doesn't match its declared
            scalar/array kind.

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
        if len(xs) != n:
            raise TypeError(f"expected {n} arguments, got {len(xs)}")

        call_args = []
        out = (ctypes.c_double * n_scalar)() if n_scalar else None
        if out is not None:
            call_args.append(out)
        results = [None] * n
        for i, (spec, x) in enumerate(zip(arg_specs, xs)):
            if spec.kind == "scalar":
                if isinstance(x, np.ndarray):
                    raise TypeError(f"argument {i} expected a scalar, got an ndarray")
                call_args.append(float(x))
            else:
                if not isinstance(x, np.ndarray):
                    raise TypeError(
                        f"argument {i} expected a numpy.ndarray, got {type(x)!r}"
                    )
                data_ptr, nitems, itemsize, shape, strides = _as_array_args(x, spec)
                d_arr = np.zeros_like(x)
                elem_ctype = _ELEM_CTYPE[spec.elem_llvm_type]
                d_data_ptr = d_arr.ctypes.data_as(ctypes.POINTER(elem_ctype))
                call_args += [
                    data_ptr,
                    d_data_ptr,
                    nitems,
                    itemsize,
                    *shape,
                    *strides,
                ]
                results[i] = d_arr

        grad_fn(*call_args)

        scalar_i = 0
        for i, spec in enumerate(arg_specs):
            if spec.kind == "scalar":
                results[i] = out[scalar_i]
                scalar_i += 1
        return tuple(results)

    def jvp(xs: tuple, seed: tuple) -> float:
        """
        Compute the forward-mode Jacobian-vector product.

        Parameters
        ----------
        xs : tuple
            The point to differentiate at, one value per argument
            (`float` or `numpy.ndarray`, matching each argument's
            declared kind).
        seed : tuple
            The tangent direction, one value per argument, matching
            `xs`'s shapes.

        Returns
        -------
        float
            The directional derivative of the underlying function at
            `xs` in direction `seed`.

        Raises
        ------
        TypeError
            If `xs` or `seed` doesn't have exactly `n_args` values.

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

        call_args = []
        for i, (spec, x, dx) in enumerate(zip(arg_specs, xs, seed)):
            if spec.kind == "scalar":
                call_args += [float(x), float(dx)]
            else:
                if not isinstance(x, np.ndarray) or not isinstance(dx, np.ndarray):
                    raise TypeError(
                        f"argument {i} expected numpy.ndarray primal and tangent"
                    )
                if dx.shape != x.shape:
                    raise ValueError(
                        f"argument {i}'s tangent shape {dx.shape} doesn't match "
                        f"its primal shape {x.shape}"
                    )
                data_ptr, nitems, itemsize, shape, strides = _as_array_args(x, spec)
                elem_ctype = _ELEM_CTYPE[spec.elem_llvm_type]
                d_data_ptr = dx.ctypes.data_as(ctypes.POINTER(elem_ctype))
                call_args += [
                    data_ptr,
                    d_data_ptr,
                    nitems,
                    itemsize,
                    *shape,
                    *strides,
                ]
        return jvp_fn(*call_args)

    return Differentiable(grad=grad, jvp=jvp, n_args=n)
