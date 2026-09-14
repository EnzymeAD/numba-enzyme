# numba-enzyme

Differentiate [Numba](https://numba.pydata.org/)-compiled Python functions via [Enzyme](https://enzyme.mit.edu/).

## Install

```bash
pip install numba-enzyme
```

### Supported platforms

| | |
|---|---|
| OS / architecture | Linux x86_64 only |
| glibc | ≥ 2.39 (e.g. Ubuntu 24.04+, Debian 13+, Fedora 39+) |
| Python | CPython 3.11, 3.12, 3.13 |

## Usage

Write an ordinary Python function and pass it to `grad`, `jvp`, `jacfwd`, or
any of the other transforms below. No type annotations are needed: like a Numba
`njit` function, each derivative callable infers the argument types from the
values it is called with and compiles one specialization per distinct set of
types, reusing it afterwards. Arguments must be floating-point scalars.
Alternatively, annotate every parameter and the return value with
`numba_enzyme.types` to fix the types up front, in which case the function is
compiled as soon as it is transformed. You can also decorate your function `f`
with `@differentiable` to expose each operation as an attribute.

```python
import math  # import numpy as np
from numba_enzyme import (
    differentiable,
    grad,
    jacfwd,
    jacfwd_column,
    jacrev,
    jacrev_row,
    jvp,
    vjp,
)
from numba_enzyme.types import Float64

def f(x, y):
    return x * y + math.cos(x * y)
    # alternatively, x * y + np.cos(x * y)

grad(f)(1.0, 2.0)               # -> (df/dx, df/dy), compiled for float64 args
jvp(f)((1.0, 2.0), (1.0, 0.0))  # -> directional derivative along (1.0, 0.0)

def f_vec(x, y):
    return x * y, x * x + y

jacfwd(f_vec)(1.0, 2.0)           # -> ((2.0, 1.0), (2.0, 1.0))
jacfwd_column(f_vec)(1.0, 2.0, 0)  # -> (2.0, 2.0)
jacrev(f_vec)(1.0, 2.0)            # -> ((2.0, 1.0), (2.0, 1.0))
jacrev_row(f_vec)(1.0, 2.0, 1)     # -> (2.0, 1.0)
vjp(f_vec)((1.0, 2.0), (0.0, 1.0)) # -> (2.0, 1.0)

@differentiable
def g(x: Float64, y: Float64) -> Float64:  # annotations fix the types up front
    return x * y + math.cos(x * y)
    # alternatively, x * y + np.cos(x * y)

g(1.0, 2.0)           # calls the original Python function directly
g.grad(1.0, 2.0)      # reverse-mode gradient, built lazily on first access
g.jvp((1.0, 2.0), (1.0, 0.0))  # forward-mode JVP
```

`grad` requires its target function to return exactly one scalar. It does not
accept tuple-valued targets. `jvp`, `jacfwd`, and `jacfwd_column` handle both
scalar and vector outputs, as do their reverse-mode counterparts `vjp`,
`jacrev`, and `jacrev_row`. CPU vector outputs use fixed-size homogeneous tuples.
Full Jacobians are returned as output-by-input tuple matrices.

### Choosing a differentiation operation

| API | Mode | Result and typical use |
|---|---|---|
| `grad` | Reverse | Gradient of a **single scalar output** in one reverse sweep. Prefer this for scalar losses, particularly with many inputs. |
| `jacfwd` | Forward | Complete Jacobian, using one sweep per input. Prefer it when there are relatively few inputs. |
| `jacfwd_column` | Forward | One Jacobian column for a selected input. Use it when only that input's effect is needed or storing the full matrix is undesirable. |
| `jacrev` | Reverse | Complete Jacobian, using one sweep per output. Prefer it when there are relatively few outputs. |
| `jacrev_row` | Reverse | One Jacobian row for a selected output. Use it when only that output's gradient is needed. |
| `jvp` | Forward | Jacobian-vector product `J @ tangent` without constructing the Jacobian. Use it for a known input direction. |
| `vjp` | Reverse | Vector-Jacobian product `cotangent @ J` without constructing the Jacobian. Use it for backpropagation from a known output cotangent. |

For a scalar-output function, `grad`, `jacfwd`, and `jacrev` have the same
values and tuple shape. Their computational paths differ: `grad` and `jacrev`
use reverse mode, while `jacfwd` uses one forward sweep per input.

### Scope

* It is still not possible to mark the arguments as active or constant.
* CPU arguments must be floating-point scalars; results may be scalar or
  fixed-size homogeneous tuples of scalars. General array inputs and outputs are not yet supported.
* Linear algebra e.g. `np.dot`, `np.linalg.norm` etc are not supported.

## License

Apache License 2.0 with LLVM Exceptions — see [LICENSE](LICENSE) and
[NOTICE](NOTICE).
