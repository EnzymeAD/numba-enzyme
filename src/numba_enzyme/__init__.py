"""
Differentiate Numba-compiled functions with Enzyme.

This package makes a pure-math Python function (annotated with
:mod:`numba_enzyme.types`) differentiable by compiling it with Numba,
synthesising an Enzyme driver for it with :mod:`llvmlite`, and running
the standalone Enzyme LLVM pass over the result. Both reverse-mode
gradients and forward-mode Jacobian-vector products are supported.

See Also
--------
numba_enzyme.core.grad : Reverse-mode gradient of a function.
numba_enzyme.core.jvp : Forward-mode Jacobian-vector product of a function.
numba_enzyme.core.differentiable : Decorator exposing ``.grad``/``.jvp``.

Examples
--------
>>> import math
>>> from numba_enzyme import differentiable
>>> from numba_enzyme.types import Float64
>>> @differentiable
... def f(x: Float64, y: Float64) -> Float64:
...     return math.sin(x) * y
>>> f(1.0, 2.0)  # doctest: +SKIP
1.682941969615793
>>> f.grad(1.0, 2.0)  # doctest: +SKIP
(1.0806046117362795, 0.8414709848078965)
"""

from importlib.metadata import version

from numba_enzyme.core import (
    Differentiable,
    differentiable,
    grad,
    jacfwd,
    jacfwd_column,
    jacrev,
    jacrev_row,
    jvp,
    vjp,
)

__version__ = version("numba-enzyme")

__all__ = [
    "Differentiable",
    "differentiable",
    "grad",
    "jacfwd",
    "jacfwd_column",
    "jacrev",
    "jacrev_row",
    "jvp",
    "vjp",
]
