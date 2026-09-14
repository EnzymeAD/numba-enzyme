"""
Public API tying lowering, driver synthesis, build, and runtime together.

Exposes the public gradient, Jacobian, JVP, and VJP transformations and the
`differentiable` decorator.

A function annotated with `numba_enzyme.types` is compiled when it is
transformed. Without annotations, it is instead specialized from the argument
types of each call to the returned callable, as a Numba ``njit`` function is.

See Also
--------
numba_enzyme.build.build : Orchestrates the compile pipeline these
    functions drive.
numba_enzyme.runtime.load : Produces the callables these functions
    return.

Examples
--------
>>> import math
>>> from numba_enzyme.core import grad
>>> from numba_enzyme.types import Float64
>>> def f(x: Float64, y: Float64) -> Float64:
...     return math.sin(x) * y
>>> grad(f)(1.0, 2.0)  # doctest: +SKIP
(1.0806046117362795, 0.8414709848078965)
"""

import functools
from collections.abc import Callable

from numba_enzyme.build import build
from numba_enzyme.lowering import is_annotated
from numba_enzyme.runtime import lazy_derivative, load


def grad(func: Callable) -> Callable:
    """
    Return a callable, computing the reverse-mode gradient of a function.

    Parameters
    ----------
    func : callable
        A scalar-output Python function. Argument and return types come from
        `numba_enzyme.types` annotations when it has them all, and are
        otherwise inferred from each call.

    Returns
    -------
    callable
        Takes the same positional arguments as `func` and returns a
        `tuple` holding the gradient with respect to each of them.

    Raises
    ------
    TypeError
        If `func` has a tuple result. An unannotated function's result is only
        known once the returned callable is called, so that callable raises
        instead.

    See Also
    --------
    jvp : The forward-mode counterpart of this function.
    differentiable : Decorator exposing this as a `.grad` attribute.

    Examples
    --------
    >>> from numba_enzyme.core import grad
    >>> from numba_enzyme.types import Float64
    >>> def f(x: Float64) -> Float64:
    ...     return x * x
    >>> grad(f)(2.0)  # doctest: +SKIP
    (4.0,)
    """
    if not is_annotated(func):
        return lazy_derivative(func, "grad")
    differentiated = load(build(func))
    if differentiated.n_outputs != 1:
        raise TypeError(
            "grad requires a scalar-output function; use jacfwd or jvp "
            "for vector outputs"
        )
    return differentiated.grad


def jvp(func: Callable) -> Callable:
    """
    Return a callable, computing the forward-mode JVP of a function.

    Parameters
    ----------
    func : callable
        A Python function with scalar parameters and a scalar or fixed
        homogeneous tuple result. Types are inferred from each call unless it
        is fully annotated with `numba_enzyme.types`.

    Returns
    -------
    callable
        Takes a `tuple` of primal values and a `tuple` of tangent
        values (both the same length as `func`'s arguments). It returns one
        `float` for a scalar output or a tuple for a tuple-valued output.

    See Also
    --------
    grad : The reverse-mode counterpart of this function.
    differentiable : Decorator exposing this as a `.jvp` attribute.

    Examples
    --------
    >>> from numba_enzyme.core import jvp
    >>> from numba_enzyme.types import Float64
    >>> def f(x: Float64) -> Float64:
    ...     return x * x
    >>> jvp(f)((2.0,), (1.0,))  # doctest: +SKIP
    4.0
    """
    if not is_annotated(func):
        return lazy_derivative(func, "jvp")
    return load(build(func)).jvp


def vjp(func: Callable) -> Callable:
    """
    Return a callable computing a reverse-mode vector-Jacobian product.

    The returned CPU callable takes a tuple of primal inputs and an output
    cotangent. The cotangent is a scalar when `func` returns a scalar and a
    tuple matching a tuple-valued result::

        def f(x: Float64, y: Float64) -> tuple[Float64, Float64]:
            return x * y, x * x + y

        vjp(f)((2.0, 3.0), (1.0, 0.0))  # (3.0, 2.0)

    Parameters
    ----------
    func : callable
        A CPU function returning a scalar or fixed homogeneous
        tuple.

    Returns
    -------
    callable
        A host callable ``(primals, cotangent)`` returning one input
        cotangent per argument.

    See Also
    --------
    jvp : The forward-mode product.
    jacrev : The complete reverse-mode Jacobian.
    """
    if not is_annotated(func):
        return lazy_derivative(func, "vjp")
    return load(build(func)).vjp


def jacrev(func: Callable) -> Callable:
    """
    Return a callable computing a whole reverse-mode Jacobian.

    The returned CPU callable takes the same positional arguments as `func`.
    A scalar result produces one derivative per input; a fixed homogeneous
    tuple result produces an output-by-input tuple matrix. Reverse mode uses
    one sweep per output component.

    Parameters
    ----------
    func : callable
        A CPU function returning a scalar or fixed homogeneous
        tuple.

    Returns
    -------
    callable
        A host callable ``(*args)`` returning a derivative tuple or Jacobian
        tuple matrix.

    See Also
    --------
    jacrev_row : One runtime-selected row of the same Jacobian.
    vjp : A reverse-mode product with an arbitrary output cotangent.
    jacfwd : The forward-mode counterpart.
    """
    if not is_annotated(func):
        return lazy_derivative(func, "jacrev")
    return load(build(func)).jacrev


def jacrev_row(func: Callable) -> Callable:
    """
    Return a callable computing one reverse-mode Jacobian row.

    The returned CPU callable takes the primal arguments followed by a
    zero-based output-row index and returns one derivative per input.

    Parameters
    ----------
    func : callable
        A CPU function returning a scalar or fixed homogeneous
        tuple.

    Returns
    -------
    callable
        A host callable ``(*args, row)`` returning the selected Jacobian row.

    See Also
    --------
    jacrev : The complete reverse-mode Jacobian.
    vjp : A reverse-mode product with an arbitrary output cotangent.
    jacfwd_column : The corresponding forward-mode column operation.
    """
    if not is_annotated(func):
        return lazy_derivative(func, "jacrev_row")
    return load(build(func)).jacrev_row


def jacfwd(func: Callable) -> Callable:
    """
    Return a callable computing a whole forward-mode Jacobian.

    For a CPU function, the returned callable takes the same
    positional arguments as `func`. A scalar result produces one partial
    derivative per argument as a `tuple`::

        def f(x: Float64, y: Float64) -> Float64:
            return x * y

        jacfwd(f)(2.0, 3.0)  # (3.0, 2.0)

    A fixed-size homogeneous tuple result produces an output-by-input tuple
    matrix.

    Parameters
    ----------
    func : callable
        A function returning a scalar or fixed homogeneous tuple.

    Returns
    -------
    callable
        A host callable ``(*args)`` returning the Jacobian as a tuple or tuple
        matrix.

    See Also
    --------
    jacfwd_column : One column of the same Jacobian, chosen at run time.
    jvp : Forward derivative of a scalar-output primal.
    """
    if not is_annotated(func):
        return lazy_derivative(func, "jacfwd")
    return load(build(func)).jacfwd


def jacfwd_column(func: Callable) -> Callable:
    """
    Return a callable computing one forward-mode Jacobian column.

    For a CPU function, the returned callable takes the primal
    arguments followed by the zero-based column index. It returns the selected
    partial derivative for a scalar result::

        def f(x: Float64, y: Float64) -> Float64:
            return x * y

        jacfwd_column(f)(2.0, 3.0, 1)  # 2.0

    For a fixed-size homogeneous tuple result it returns that column as a
    tuple, with one derivative per output component.

    Parameters
    ----------
    func : callable
        A function returning a scalar or fixed homogeneous tuple.

    Returns
    -------
    callable
        A host callable ``(*args, index)`` returning one scalar or tuple
        column.

    See Also
    --------
    jacfwd : The whole Jacobian, one sweep per column.
    jvp : Forward derivative of a scalar-output primal.
    """
    if not is_annotated(func):
        return lazy_derivative(func, "jacfwd_column")
    return load(build(func)).jacfwd_column


def _cached_transform(transform: Callable) -> functools.cached_property:
    """
    Build a `Differentiable` attribute applying one transformation.

    Parameters
    ----------
    transform : callable
        Module-level transformation, such as `grad`, applied to the wrapped
        function the first time the attribute is read.

    Returns
    -------
    functools.cached_property
        Attribute returning ``transform(func)``, built once and cached.

    See Also
    --------
    Differentiable : The class these attributes are defined on.

    Examples
    --------
    >>> from numba_enzyme.core import _cached_transform, grad
    >>> type(_cached_transform(grad)).__name__
    'cached_property'
    """
    name = transform.__name__

    def derivative(self):
        """
        Return this transformation's callable for the wrapped function.

        Parameters
        ----------
        self : Differentiable
            Wrapper holding the function to transform.

        Returns
        -------
        callable
            The transformed callable, built once and cached thereafter.

        See Also
        --------
        _cached_transform : Builds the attribute this backs.
        """
        return transform(self._func)

    derivative.__name__ = name
    derivative.__doc__ = (
        f"Return the `{name}` callable for the wrapped function.\n\n"
        "Built once on first access and cached thereafter.\n\n"
        "Returns\n-------\ncallable\n"
        f"    Equivalent to ``numba_enzyme.core.{name}(self)``.\n\n"
        f"See Also\n--------\nnumba_enzyme.core.{name} : "
        "The standalone equivalent this wraps.\n"
    )
    return functools.cached_property(derivative)


class Differentiable:
    """
    Wrap a Python function with lazily-built derivative callables.

    Calling an instance runs the original Python code directly. Its derivative
    callables are built and cached on first access, so decorating a function
    costs nothing until it is actually differentiated. The available
    attributes are ``.grad``, ``.jvp``, ``.vjp``, ``.jacfwd``,
    ``.jacfwd_column``, ``.jacrev``, and ``.jacrev_row``.

    Parameters
    ----------
    func : callable
        A Python function with scalar parameters and a scalar or homogeneous
        tuple result, optionally annotated with `numba_enzyme.types` classes.

    See Also
    --------
    differentiable : Constructs a `Differentiable` instance.

    Examples
    --------
    >>> from numba_enzyme.core import Differentiable
    >>> from numba_enzyme.types import Float64
    >>> def f(x: Float64) -> Float64:
    ...     return x * x
    >>> d = Differentiable(f)
    >>> d(2.0)
    4.0
    >>> d.grad(2.0)  # doctest: +SKIP
    (4.0,)
    """

    def __init__(self, func: Callable):
        functools.update_wrapper(self, func)
        self._func = func

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    # Every mode is the matching module-level transformation applied lazily,
    # so decorating stays free until a derivative is actually asked for.
    grad = _cached_transform(grad)
    jvp = _cached_transform(jvp)
    vjp = _cached_transform(vjp)
    jacfwd = _cached_transform(jacfwd)
    jacfwd_column = _cached_transform(jacfwd_column)
    jacrev = _cached_transform(jacrev)
    jacrev_row = _cached_transform(jacrev_row)


def differentiable(func: Callable) -> Differentiable:
    """
    Mark a function as differentiable.

    Parameters
    ----------
    func : callable
        A Python function with scalar parameters and a scalar or homogeneous
        tuple result, optionally annotated with `numba_enzyme.types` classes.

    Returns
    -------
    Differentiable
        A wrapper exposing every public derivative transformation as a cached
        attribute alongside normal calls to `func` itself.

    See Also
    --------
    Differentiable : The wrapper this decorator returns.
    grad : The standalone equivalent of `.grad`.
    jvp : The standalone equivalent of `.jvp`.
    jacfwd : The standalone equivalent of `.jacfwd`.
    jacfwd_column : The standalone equivalent of `.jacfwd_column`.
    vjp : The standalone equivalent of `.vjp`.
    jacrev : The standalone equivalent of `.jacrev`.
    jacrev_row : The standalone equivalent of `.jacrev_row`.

    Examples
    --------
    >>> from numba_enzyme.core import differentiable
    >>> from numba_enzyme.types import Float64
    >>> @differentiable
    ... def f(x: Float64) -> Float64:
    ...     return x * x
    >>> f(2.0)
    4.0
    >>> f.grad(2.0)  # doctest: +SKIP
    (4.0,)
    """
    return Differentiable(func)
