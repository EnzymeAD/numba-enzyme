# numba-enzyme

Differentiate [Numba](https://numba.pydata.org/)-compiled Python functions via [Enzyme](https://enzyme.mit.edu/).

## Install

```bash
pip install numba-enzyme
```

### Supported platforms

| | |
|---|---|
| OS / architecture | Linux (`x86_64`, `aarch64`), macOS (`arm64` Apple Silicon, macOS 12.0+) |
| Linux glibc | ≥ 2.35 (e.g. Ubuntu 22.04+, Debian 12+, RHEL 9+, Fedora 36+) |
| Python | CPython 3.11, 3.12, 3.13, 3.14, 3.15 |

## Usage

Arguments of the function must be annotated by the `numba_enzyme.types`. This is required to compile
the function to a Numba `cfunc`. Then use `grad`, `jvp` and pass the inputs. Also you can decorate
your function `f` with `@differentiable`, and call `f.grad` or `f.jvp` to get the gradient or
Jacobian-vector product, respectively.

```python
import math  # import numpy as np
from numba_enzyme.core import grad, jvp, differentiable
from numba_enzyme.types import Float64


def f(x: Float64, y: Float64) -> Float64:
    return x * y + math.cos(x * y)
    # alternatively, x * y + np.cos(x * y)


grad(f)(1.0, 2.0)  # -> (df/dx, df/dy)
jvp(f)((1.0, 2.0), (1.0, 0.0))  # -> directional derivative along (1.0, 0.0)


@differentiable
def g(x: Float64, y: Float64) -> Float64:
    return x * y + math.cos(x * y)
    # alternatively, x * y + np.cos(x * y)


g(1.0, 2.0)  # calls the original Python function directly
g.grad(1.0, 2.0)  # reverse-mode gradient, built lazily on first access
g.jvp((1.0, 2.0), (1.0, 0.0))  # forward-mode JVP
```

## Building from source

`numba-enzyme` uses [Bazel](https://bazel.build/) (via [Bazelisk](https://github.com/bazelbuild/bazelisk)) to hermetically build LLVM, Enzyme, and Python wheels across Linux and macOS:

```bash
# Run the test suite (defaults to HERMETIC_PYTHON_VERSION=3.11; supports 3.11..3.15)
HERMETIC_PYTHON_VERSION=3.13 bazel test //tests:python_tests

# Build a platform wheel into bazel-bin/
bazel build //:wheel
```

For local editable development with `uv`:

```bash
bazel build //src/numba_enzyme:libNumbaEnzyme.so
uv run pytest -v
```

### Scope
* It is still not possible to mark the arguments as active or constant.
* Return values must be scalar; scalar and C-contiguous `numpy.ndarray` inputs (`Array1D`, `Array2D`, `ArrayND`) are supported.
* Linear algebra e.g. `np.dot`, `np.linalg.norm` etc are not supported.

## License

Apache License 2.0 with LLVM Exceptions — see [LICENSE](LICENSE) and
[NOTICE](NOTICE).
