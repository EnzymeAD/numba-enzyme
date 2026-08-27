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

Arguments of the function must be annotated by the `numba_enzyme.types`. This is required to compile
the function to a Numba `cfunc`. Then use `grad`, `jvp` and pass the inputs. Also you can decorate
your function `f` with `@differentiable`, and call `f.grad` or `f.jvp` to get the gradient or
Jacobian-vector product, respectively.

```python
import math # import numpy as np
from numba_enzyme.core import grad, jvp, differentiable
from numba_enzyme.types import Float64

def f(x: Float64, y: Float64) -> Float64:
    return x * y + math.cos(x * y)
    # alternatively, x * y + np.cos(x * y)

grad(f)(1.0, 2.0)               # -> (df/dx, df/dy)
jvp(f)((1.0, 2.0), (1.0, 0.0))  # -> directional derivative along (1.0, 0.0)

@differentiable
def g(x: Float64, y: Float64) -> Float64:
    return x * y + math.cos(x * y)
    # alternatively, x * y + np.cos(x * y)

g(1.0, 2.0)           # calls the original Python function directly
g.grad(1.0, 2.0)      # reverse-mode gradient, built lazily on first access
g.jvp((1.0, 2.0), (1.0, 0.0))  # forward-mode JVP
```


### Scope
* It is still not possible to mark the arguments as active or conastant.
* The arguments and the return value can be of scalar type only at the moment.
* Linear algebra e.g. `np.dot`, `np.linalg.norm` etc are not supported.

## License

Apache License 2.0 with LLVM Exceptions — see [LICENSE](LICENSE) and
[NOTICE](NOTICE).
