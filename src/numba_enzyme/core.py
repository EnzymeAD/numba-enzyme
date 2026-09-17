"""
Public API tying lowering, driver synthesis, build, and runtime together.

Exposes the public gradient, Jacobian, JVP, and VJP transformations and the
`differentiable` decorator.

A CPU function annotated with `numba_enzyme.types` is compiled when it is
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


def _differentiate(func, mode, signature, cc):
    """
    Return one derivative callable for a CPU or CUDA function.

    Parameters
    ----------
    func : callable
        Annotated CPU function or CUDA device dispatcher to differentiate.
    mode : str
        Name of the derivative, such as ``"grad"`` or ``"jacfwd"``.
    signature : object or None
        Concrete CUDA specialization, or `None`.
    cc : tuple of int or None
        CUDA compute capability, or `None`.

    Returns
    -------
    callable
        The requested host or CUDA device callable.

    Raises
    ------
    TypeError
        If CUDA-only options are passed for a CPU function, or `grad` is asked
        for a vector result.

    See Also
    --------
    numba_enzyme.cuda.lazy_cuda_derivative : Builds CUDA derivatives lazily.
    """
    from numba_enzyme.cuda import is_cuda_device_function, lazy_cuda_derivative

    # Composing a derivative with itself: jvp(jvp(f)) is the second-order
    # forward sweep. The chain is recorded rather than applied, because a
    # derivative that has already been built is an external device symbol with
    # no body for Enzyme to differentiate; the whole chain is emitted as
    # definitions in one module when the outermost call site is compiled.
    base = getattr(func, "_numba_enzyme_primal", None)
    depth = 1
    if base is not None:
        inner_mode = getattr(func, "_numba_enzyme_mode", None)
        if inner_mode != "jvp":
            raise TypeError(
                f"{mode} does not compose over {inner_mode}; only a forward "
                "directional derivative (jvp) can be an inner level"
            )
        depth = func._numba_enzyme_depth + 1
        func = base

    if is_cuda_device_function(func):
        return lazy_cuda_derivative(func, mode, signature=signature, cc=cc, depth=depth)
    if depth != 1:
        raise TypeError("composing derivatives is only supported for CUDA")
    if signature is not None or cc is not None:
        raise TypeError("signature and cc are only valid for CUDA device functions")
    if not is_annotated(func):
        return lazy_derivative(func, mode)
    differentiated = load(build(func))
    if mode == "grad" and differentiated.n_outputs != 1:
        raise TypeError(
            "grad requires a scalar-output function; use jacfwd or jvp "
            "for vector outputs"
        )
    return getattr(differentiated, mode)


def grad(func: Callable, *, signature=None, cc=None) -> Callable:
    """
    Return a callable, computing the reverse-mode gradient of a function.

    Parameters
    ----------
    func : callable
        A scalar-output Python function or a ``@cuda.jit(device=True)``
        dispatcher. Argument and return types come from `numba_enzyme.types`
        annotations when a CPU function has them all, and are otherwise
        inferred from each call.
    signature : numba_cuda_mlir.typing.Signature, optional
        Concrete specialization to differentiate for a CUDA device function.
        This optionally constrains lazy call-site specialization and is invalid
        for CPU functions.
    cc : tuple of int, optional
        CUDA compute capability as ``(major, minor)``. If omitted, the backend
        uses the current device's compute capability.
        This option is invalid for CPU functions.

    Returns
    -------
    callable
        Takes the same positional arguments as `func` and returns a
        `tuple` holding the gradient with respect to each of them. For CUDA,
        this is a device callable intended for use inside CUDA-compiled code.

    Raises
    ------
    TypeError
        If CUDA-only options are passed for a CPU function, or a CPU function
        has a tuple result. An unannotated function's result is only known
        once the returned callable is called, so that callable raises instead.

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
    return _differentiate(func, "grad", signature, cc)


def jvp(func: Callable, *, signature=None, cc=None) -> Callable:
    """
    Return a callable, computing the forward-mode JVP of a function.

    Parameters
    ----------
    func : callable
        A Python function with scalar parameters and a scalar or fixed
        homogeneous tuple result, or a ``@cuda.jit(device=True)`` dispatcher.
        Types are inferred from each call unless a CPU function is fully
        annotated with `numba_enzyme.types`.
    signature : numba_cuda_mlir.typing.Signature, optional
        Concrete specialization to differentiate for a CUDA device function.
        This optionally constrains lazy call-site specialization and is invalid
        for CPU functions.
    cc : tuple of int, optional
        CUDA compute capability as ``(major, minor)``. If omitted, the backend
        uses the current device's compute capability.
        This option is invalid for CPU functions.

    Returns
    -------
    callable
        Takes a `tuple` of primal values and a `tuple` of tangent
        values (both the same length as `func`'s arguments). It returns one
        `float` for a scalar CPU output, a tuple for a tuple-valued CPU output,
        or one scalar for CUDA. The CUDA result is a device callable intended
        for use inside CUDA-compiled code.

    For a tuple-returning CUDA primal the call takes the array shape
    ``(tangent, *args, *directions)``, one mirrored direction set per sweep;
    several sets write a matrix, one row each, and `jacfwd` is that same loop
    with the identity supplied internally.

    A direction the compiler can see folds. The CUDA derivative links as LTO
    IR, so nvJitLink inlines it before constant propagation: a unit direction
    spelled out as literals at the call site kills the tangent arithmetic of
    every zero component and collapses the sweep to the one column that
    survives. Asking for a single column therefore needs no separate endpoint,
    only a literal seed.

    A CUDA derivative is also a valid primal: ``jvp(jvp(f))`` is the
    second-order forward sweep, and every endpoint composes over a `jvp`. The
    chain is recorded rather than applied -- an already-built derivative is an
    external symbol with no body for Enzyme to differentiate -- and emitted as
    definitions, one Enzyme run per stage.

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
    return _differentiate(func, "jvp", signature, cc)


def vjp(func: Callable, *, signature=None, cc=None) -> Callable:
    """
    Return a callable computing a reverse-mode vector-Jacobian product.

    A CPU or scalar-output CUDA callable takes a tuple of primal inputs and an
    output cotangent; a tuple-returning CUDA primal uses the array call shape
    instead. The cotangent is a scalar when `func` returns a scalar
    and a tuple matching a tuple-valued CPU result::

        def f(x: Float64, y: Float64) -> tuple[Float64, Float64]:
            return x * y, x * x + y

        vjp(f)((2.0, 3.0), (1.0, 0.0))  # (3.0, 2.0)

    Parameters
    ----------
    func : callable
        A CPU function returning a scalar or fixed homogeneous
        tuple, or a CUDA device dispatcher.
    signature : numba_cuda_mlir.typing.Signature, optional
        Concrete CUDA specialization every call site must resolve to. CUDA
        argument types are otherwise inferred at each call site. Invalid for
        CPU functions.
    cc : tuple of int, optional
        CUDA compute capability. Invalid for CPU functions.

    Returns
    -------
    callable
        A host or CUDA device callable computing one input cotangent per
        argument. Tuple-returning CUDA primals use the array call shape
        documented in the README.

    Raises
    ------
    TypeError
        If CUDA-only options are passed for a CPU function.

    See Also
    --------
    jvp : The forward-mode product.
    jacrev : The complete reverse-mode Jacobian.
    """
    return _differentiate(func, "vjp", signature, cc)


def jacrev(func: Callable, *, signature=None, cc=None) -> Callable:
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
        tuple, or a CUDA device dispatcher.
    signature : numba_cuda_mlir.typing.Signature, optional
        Concrete CUDA specialization every call site must resolve to. CUDA
        argument types are otherwise inferred at each call site. Invalid for
        CPU functions.
    cc : tuple of int, optional
        CUDA compute capability. Invalid for CPU functions.

    Returns
    -------
    callable
        A host callable ``(*args)`` returning a derivative tuple or Jacobian
        tuple matrix.

    Raises
    ------
    TypeError
        If CUDA-only options are passed for a CPU function.

    See Also
    --------
    vjp : A reverse-mode product with an arbitrary output cotangent.
    jacfwd : The forward-mode counterpart.
    """
    return _differentiate(func, "jacrev", signature, cc)


def jacfwd(func: Callable, *, signature=None, cc=None) -> Callable:
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

    A CUDA primal returns a homogeneous tuple, which Numba-CUDA-MLIR lowers to
    a struct returned by value. One forward sweep gives a whole Jacobian
    column, and one sweep per input fills the complete matrix, which the
    derivative writes into a caller-owned ``n_out`` by ``n_args`` array::

        jac = jacfwd(f)
        # inside CUDA-compiled code:
        jac(jacobian, x0, x1)

    On a GPU ``jacobian`` costs ``n_out * n_args`` per thread, which stops
    being viable well before the dimensions a sweep at a time handles
    comfortably; prefer `jvp` when the whole matrix need not exist at once. A
    unit direction spelled out at the call site folds, so asking for one
    column that way costs one column.

    Parameters
    ----------
    func : callable
        A CPU function returning a scalar or fixed homogeneous
        tuple, or a
        ``@cuda.jit(device=True)`` dispatcher of the shape
        ``UniTuple(dtype, n)(x0, ..., xn)``.
    signature : numba_cuda_mlir.typing.Signature, optional
        Concrete CUDA specialization every call site must resolve to. CUDA
        argument types are otherwise inferred at each call site. Invalid for
        CPU functions.
    cc : tuple of int, optional
        CUDA compute capability as ``(major, minor)``. Defaults to the current
        device's.

    Returns
    -------
    callable
        For CPU, a host callable ``(*args)`` returning the Jacobian as a tuple
        or tuple matrix. For CUDA, a device callable ``(jacobian, *args)``
        returning nothing.

    Raises
    ------
    TypeError
        If CUDA-only options are passed for a CPU function.

    See Also
    --------
    jvp : Forward derivative of a scalar-output primal.
    """
    return _differentiate(func, "jacfwd", signature, cc)


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
    attributes are ``.grad``, ``.jvp``, ``.vjp``, ``.jacfwd``, and
    ``.jacrev``.

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
    jacrev = _cached_transform(jacrev)


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
    vjp : The standalone equivalent of `.vjp`.
    jacrev : The standalone equivalent of `.jacrev`.

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
