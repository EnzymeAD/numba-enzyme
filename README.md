# numba-enzyme

Differentiate [Numba](https://numba.pydata.org/)-compiled Python functions
with the standalone [Enzyme](https://enzyme.mit.edu/) LLVM plugin.

Originally developed by Yousof Mardoukhi on behalf of [Qruise GmbH](https://qruise.com).

## Install

```bash
pip install numba-enzyme
```

Wheels bundle their own copy of `clang`/`opt`/`llvm-link`, the Enzyme LLVM
pass plugin, and the C-runtime pieces needed to link the compiled kernels —
nothing beyond `pip install` is required on the target machine.

### Supported platforms

| | |
|---|---|
| OS / architecture | Linux x86_64 only |
| glibc | ≥ 2.39 (e.g. Ubuntu 24.04+, Debian 13+, Fedora 39+) |
| Python | CPython 3.11, 3.12, 3.13 |

Not supported: macOS, Windows, musl-based distros (Alpine), ARM/aarch64,
i686, PyPy, or free-threaded builds.

## Usage

Annotate each parameter and the return value with a type from
`numba_enzyme.types`, then use `grad`, `jvp`, or the `@differentiable`
decorator:

```python
import math
from numba_enzyme.core import grad, jvp, differentiable
from numba_enzyme.types import Float64

def f(x: Float64, y: Float64) -> Float64:
    return x * y + math.cos(x * y)

grad(f)(1.0, 2.0)              # -> (df/dx, df/dy)
jvp(f)((1.0, 2.0), (1.0, 0.0))  # -> directional derivative along (1.0, 0.0)

@differentiable
def g(x: Float64, y: Float64) -> Float64:
    return x * y + math.cos(x * y)

g(1.0, 2.0)          # calls the original Python function directly
g.grad(1.0, 2.0)      # reverse-mode gradient, built lazily on first access
g.jvp((1.0, 2.0), (1.0, 0.0))  # forward-mode JVP
```

`grad` returns the gradient with respect to every argument, as a tuple.
`jvp` takes a tuple of primal values and a tuple of tangent (seed) values
and returns a single float — the Jacobian-vector product.

### Scope

Currently supports pure real scalar math (`+`, `*`, `**`, `math.sin`/`cos`/`exp`,
...) for functions of the shape `f(x0: T0, ..., xN: TN) -> TR`, both
forward- and reverse-mode. Arrays/reductions, `np.linalg`/BLAS, and
`complex128` are not supported yet.

## License

Apache License 2.0 with LLVM Exceptions — see [LICENSE](LICENSE) and
[NOTICE](NOTICE).
