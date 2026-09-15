"""
Load a built shared object and expose it as plain Python callables.

Wraps a `numba_enzyme.build.BuiltKernel`'s ``jvp_<entry>``/``vjp_<entry>``
symbols with :mod:`ctypes` and derives every other mode from them: the
gradient and the reverse-mode Jacobians are vector-Jacobian products under
unit cotangents, and the forward-mode Jacobians are JVPs under unit tangents.
``vjp_<entry>`` is void and writes through one explicit output pointer per
argument, uniformly for every arity (see `numba_enzyme.driver`).
For a scalar result, ``jvp_<entry>`` returns a bare scalar. For a tuple result,
it writes the output tangent through an explicit pointer.

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
import operator
from collections.abc import Callable
from dataclasses import dataclass

from numba_enzyme.build import BuiltKernel

_CTYPES_SCALAR_TYPE = {
    "double": ctypes.c_double,
    "float": ctypes.c_float,
    "i32": ctypes.c_int32,
    "i64": ctypes.c_int64,
}


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
    jacfwd_column : callable
        Computes one forward-mode Jacobian column. Takes `n_args` primal values
        followed by a column index and returns a scalar or output tuple.
    vjp : callable
        Computes a reverse-mode vector-Jacobian product from primal values and
        an output cotangent.
    jacrev : callable
        Computes the full Jacobian in reverse mode, one output row per sweep.
    jacrev_row : callable
        Computes one runtime-selected Jacobian row in reverse mode.
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
    jacfwd_column: Callable[..., float | tuple[float, ...]]
    vjp: Callable[[tuple[float, ...], object], tuple[float, ...]]
    jacrev: Callable[..., tuple]
    jacrev_row: Callable[..., tuple[float, ...]]
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


def _split_index(args, n_args: int, limit: int, kind: str, noun: str):
    """
    Split primal values from a trailing row or column index.

    Parameters
    ----------
    args : tuple
        The primal values followed by exactly one index.
    n_args : int
        Number of primal values expected.
    limit : int
        Exclusive upper bound on the index.
    kind : str
        What the index selects, ``"row"`` or ``"column"``, for error messages.
    noun : str
        What `limit` counts, for error messages.

    Returns
    -------
    xs : tuple of float
        The primal values.
    index : int
        The validated index.

    Raises
    ------
    TypeError
        If the argument count is wrong or the index is not integral.
    IndexError
        If the index is out of range.

    See Also
    --------
    _unit : Turns the returned index into a seed.

    Examples
    --------
    >>> from numba_enzyme.runtime import _split_index
    >>> _split_index((2.0, 3.0, 1), 2, 2, "column", "arguments")
    ((2.0, 3.0), 1)
    """
    if len(args) != n_args + 1:
        raise TypeError(
            f"expected {n_args} primal arguments and one {kind} index, got {len(args)}"
        )
    *xs, index = args
    try:
        index = operator.index(index)
    except TypeError:
        raise TypeError(f"{kind} index must be an integer") from None
    if not 0 <= index < limit:
        raise IndexError(f"{kind} index {index} is out of range for {limit} {noun}")
    return tuple(xs), index


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
    arg_ctypes = [_CTYPES_SCALAR_TYPE[arg_type] for arg_type in built.arg_types]
    return_ctype = _CTYPES_SCALAR_TYPE[built.return_type]

    vjp_fn = getattr(lib, built.vjp_symbol)
    vjp_fn.restype = None
    vjp_fn.argtypes = [
        *[ctypes.POINTER(ctype) for ctype in arg_ctypes],
        ctypes.POINTER(return_ctype),
        *arg_ctypes,
    ]

    jvp_fn = getattr(lib, built.jvp_symbol)
    interleaved_ctypes = [ctype for item in arg_ctypes for ctype in (item, item)]
    if m == 1:
        jvp_fn.restype = return_ctype
        jvp_fn.argtypes = interleaved_ctypes
    else:
        jvp_fn.restype = None
        jvp_fn.argtypes = [ctypes.POINTER(return_ctype), *interleaved_ctypes]

    def jvp(
        xs: tuple[float, ...], seed: tuple[float, ...]
    ) -> float | tuple[float, ...]:
        """
        Compute the forward-mode Jacobian-vector product.

        Parameters
        ----------
        xs : tuple of float
            The point to differentiate at, one value per argument.
        seed : tuple of float
            The tangent direction, one value per argument.

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
        interleaved = [value for pair in zip(xs, seed) for value in pair]
        if m == 1:
            return jvp_fn(*interleaved)
        out = (return_ctype * m)()
        jvp_fn(out, *interleaved)
        return tuple(out)

    def vjp(
        xs: tuple[float, ...], cotangent: float | tuple[float, ...]
    ) -> tuple[float, ...]:
        """
        Compute a reverse-mode vector-Jacobian product.

        Parameters
        ----------
        xs : tuple of float
            The point to differentiate at, one value per input argument.
        cotangent : float or tuple of float
            Scalar seed for a scalar result or one seed per vector component.

        Returns
        -------
        tuple of float
            Input cotangent, one value per primal argument.

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

        outputs = [ctype() for ctype in arg_ctypes]
        vjp_fn(
            *[ctypes.byref(output) for output in outputs],
            (return_ctype * m)(*cotangents),
            *xs,
        )
        return tuple(output.value for output in outputs)

    def grad(*xs: float) -> tuple[float, ...]:
        """
        Compute the reverse-mode gradient.

        Parameters
        ----------
        *xs : float
            The point(s) to differentiate at.

        Returns
        -------
        tuple of float
            The gradient with respect to each argument.

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
                "for vector outputs"
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
            For vector output, an output-by-input tuple matrix.

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
        if len(xs) != n:
            raise TypeError(f"expected {n} arguments, got {len(xs)}")
        columns = tuple(jvp(xs, _unit(n, column)) for column in range(n))
        if m == 1:
            return columns
        return tuple(
            tuple(columns[column][row] for column in range(n)) for row in range(m)
        )

    def jacfwd_column(*args: float) -> float | tuple[float, ...]:
        """
        Compute one column of the forward-mode Jacobian.

        Parameters
        ----------
        *args : float
            The primal values followed by a zero-based integer column index.

        Returns
        -------
        float or tuple of float
            The partial derivative for the selected input column. A
            vector-output function returns one value per output component.

        Raises
        ------
        TypeError
            If the number of arguments is wrong or the index is not integral.
        IndexError
            If the column index is out of range.

        Examples
        --------
        >>> from numba_enzyme.build import build
        >>> from numba_enzyme.runtime import load
        >>> from numba_enzyme.types import Float64
        >>> def f(x: Float64) -> Float64:
        ...     return x * x
        >>> load(build(f)).jacfwd_column(2.0, 0)  # doctest: +SKIP
        4.0
        """
        xs, column = _split_index(args, n, n, "column", "arguments")
        return jvp(xs, _unit(n, column))

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
        if len(xs) != n:
            raise TypeError(f"expected {n} arguments, got {len(xs)}")
        if m == 1:
            return vjp(xs, 1.0)
        return tuple(vjp(xs, _unit(m, row)) for row in range(m))

    def jacrev_row(*args: float) -> tuple[float, ...]:
        """
        Compute one row of the reverse-mode Jacobian.

        Parameters
        ----------
        *args : float
            The primal values followed by a zero-based integer row index.

        Returns
        -------
        tuple of float
            One derivative per primal input for the selected output row.

        Raises
        ------
        TypeError
            If the number of arguments is wrong or the index is not integral.
        IndexError
            If the row index is out of range.

        Examples
        --------
        >>> from numba_enzyme.build import build
        >>> from numba_enzyme.runtime import load
        >>> from numba_enzyme.types import Float64
        >>> def f(x: Float64) -> Float64:
        ...     return x * x
        >>> load(build(f)).jacrev_row(2.0, 0)  # doctest: +SKIP
        (4.0,)
        """
        xs, row = _split_index(args, n, m, "row", "outputs")
        return vjp(xs, 1.0 if m == 1 else _unit(m, row))

    return Differentiable(
        grad=grad,
        jvp=jvp,
        jacfwd=jacfwd,
        jacfwd_column=jacfwd_column,
        vjp=vjp,
        jacrev=jacrev,
        jacrev_row=jacrev_row,
        n_args=n,
        n_outputs=m,
    )
