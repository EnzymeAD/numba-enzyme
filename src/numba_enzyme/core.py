"""
Public API tying lowering, driver synthesis, build, and runtime together.

Exposes `grad`, `jvp`, and the `differentiable` decorator.

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
from numba_enzyme.runtime import load


def grad(func: Callable) -> Callable[..., tuple[float, ...]]:
    """
    Return a callable, computing the reverse-mode gradient of a function.

    Parameters
    ----------
    func : callable
        A Python function whose parameters and return value are each
        annotated with a `numba_enzyme.types` class.

    Returns
    -------
    callable
        Takes the same positional arguments as `func` and returns a
        `tuple` holding the gradient with respect to each of them. Raises
        `TypeError` if `func` returns a tuple.

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
    return load(build(func)).grad


def jvp(func: Callable) -> Callable[[tuple, tuple], float | tuple]:
    """
    Return a callable, computing the forward-mode JVP of a function.

    Parameters
    ----------
    func : callable
        A Python function whose parameters are each annotated with a
        `numba_enzyme.types` class, and whose return value is annotated
        with one or a fixed-size homogeneous tuple of them.

    Returns
    -------
    callable
        Takes a `tuple` of primal values and a `tuple` of tangent
        values (both the same length as `func`'s arguments) and returns
        the Jacobian-vector product as a single `float`, or a `tuple`
        with one value per output if `func` returns a tuple.

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
    return load(build(func)).jvp


class Differentiable:
    """
    Wrap a Python function with lazily-built ``.grad``/``.jvp``.

    Calling an instance runs the original Python code directly; `.grad`
    and `.jvp` are built (and cached) on first access, so decorating a
    function costs nothing until it is actually differentiated.

    Parameters
    ----------
    func : callable
        A Python function whose parameters and return value are each
        annotated with a `numba_enzyme.types` class.

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

    @functools.cached_property
    def grad(self) -> Callable[..., tuple[float, ...]]:
        """
        Return the reverse-mode gradient callable.

        Built once on the first access. Cached thereafter.

        Returns
        -------
        callable
            Equivalent to ``numba_enzyme.core.grad(self)``.

        See Also
        --------
        numba_enzyme.core.grad : The standalone equivalent this wraps.

        Examples
        --------
        >>> from numba_enzyme.core import Differentiable
        >>> from numba_enzyme.types import Float64
        >>> def f(x: Float64) -> Float64:
        ...     return x * x
        >>> Differentiable(f).grad(2.0)  # doctest: +SKIP
        (4.0,)
        """
        return grad(self._func)

    @functools.cached_property
    def jvp(self) -> Callable[[tuple, tuple], float]:
        """
        Return the forward-mode JVP callable.

        Built once on first access. Cached thereafter.

        Returns
        -------
        callable
            Equivalent to ``numba_enzyme.core.jvp(self)``.

        See Also
        --------
        numba_enzyme.core.jvp : The standalone equivalent this wraps.

        Examples
        --------
        >>> from numba_enzyme.core import Differentiable
        >>> from numba_enzyme.types import Float64
        >>> def f(x: Float64) -> Float64:
        ...     return x * x
        >>> Differentiable(f).jvp((2.0,), (1.0,))  # doctest: +SKIP
        4.0
        """
        return jvp(self._func)


def differentiable(func: Callable) -> Differentiable:
    """
    Mark a function as differentiable.

    Parameters
    ----------
    func : callable
        A Python function whose parameters and return value are each
        annotated with a `numba_enzyme.types` class.

    Returns
    -------
    Differentiable
        A wrapper exposing `.grad`/`.jvp` alongside normal calls to
        `func` itself.

    See Also
    --------
    Differentiable : The wrapper this decorator returns.
    grad : The standalone equivalent of `.grad`.
    jvp : The standalone equivalent of `.jvp`.

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
