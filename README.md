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

### CUDA device functions

Install the optional Numba-CUDA-MLIR integration and a CUDA toolkit:

```bash
pip install 'numba-enzyme[cuda]'
```

An `@cuda.jit(device=True)` function can be differentiated on the host without
adding Python type annotations. The result is another device callable that
Numba-CUDA-MLIR automatically specializes and links into every kernel that uses
it:

```python
from numba_cuda_mlir import cuda, types
from numba_enzyme import grad, jvp

@cuda.jit(device=True)
def f(x, y):
    return x * y + x * x

df = grad(f)
jf = jvp(f)

@cuda.jit
def use_derivatives(xs, ys, gradients, directional_derivatives):
    i = cuda.grid(1)
    if i < xs.size:
        dx, dy = df(xs[i], ys[i])
        gradients[i, 0] = dx
        gradients[i, 1] = dy
        directional_derivatives[i] = jf(
            (xs[i], ys[i]), (1.0, 0.0)
        )
```

For CUDA, every transform specializes lazily from the concrete argument types
at each call site, so none needs a signature. `grad` and `jvp` differentiate an
ordinary scalar-returning device function. A primal with several outputs
**returns a homogeneous tuple**, which Numba-CUDA-MLIR lowers to an LLVM struct
returned by value: forward modes take Enzyme's tangent struct directly, and
reverse modes differentiate an internal scalarisation `sum_k w_k * f_k(x)` with
the weights inactive, since Enzyme does not accept an aggregate differential
return. Nothing is staged through an output array on either side of the call.
One forward sweep gives a whole Jacobian column:

```python
import numba
from numba_enzyme import jacfwd, jacfwd_column

@cuda.jit(device=True)
def f_tuple(x, y):
    return (x * y, x * x + y)

whole = jacfwd(f_tuple)
one = jacfwd_column(f_tuple)

@cuda.jit
def use_jacfwd(xs, ys):
    i = cuda.grid(1)
    if i < xs.size:
        column = cuda.local.array(2, numba.float64)
        jac = cuda.local.array((2, 2), numba.float64)

        whole(jac, xs[i], ys[i])              # the whole matrix
        one(column, xs[i], ys[i], 1)          # just column 1
```

**Tuple arguments.** A primal may also *take* homogeneous tuples. Every
derivative call then mirrors the primal's own argument list, with each tuple
argument supplied as a contiguous array of its elements — so a wide primal is
called with a fixed handful of arguments rather than one per scalar. The entry
point loads the elements itself, before the Enzyme marker, so the function
being differentiated is the same flat-scalar one either way:

```python
from numba_cuda_mlir import types

@cuda.jit(device=True)
def rhs(ys, t, ps):                        # 48 states, 1 parameter
    return (ys[0] * ps[0], ys[1] + t, ...)

signature = types.UniTuple(types.float64, 48)(
    types.UniTuple(types.float64, 48), types.float64, types.UniTuple(types.float64, 1)
)
one = jacfwd_column(rhs, signature=signature)

# inside CUDA-compiled code, five arguments whatever the tuple lengths are:
one(column, ys[i], t, ps[i], k)
```

An array cannot say how long the tuple it stands for is, so a primal with
tuple arguments needs an explicit `signature`; the call only has to get
array-ness right. The column index runs over the *flattened* arguments, so in
the example above `k = 48` seeds `t` and yields `d rhs / d t`.

`vjp`, `jacrev`, and `jacrev_row` accept either a scalar-returning or a
tuple-returning primal, and tell them apart by how many arguments the call
passes. They use the same output-by-input Jacobian layout:

```python
from numba_enzyme import jacrev, jacrev_row, vjp

reverse = jacrev(f_tuple)
row = jacrev_row(f_tuple)
product = vjp(f_tuple)

@cuda.jit
def use_reverse(x, y, cotangent, jacobian, input_cotangent):
    # Complete Jacobian and one selected output row.
    reverse(jacobian, x, y)
    row(input_cotangent, x, y, 1)

    # cotangent @ J.
    product(cotangent, input_cotangent, x, y)
```

`jacobian` has shape `(n_out, n_args)`. `jacrev_row` fills the selected row
into an `n_args` `input_cotangent` buffer. `vjp` accepts an `n_out` `cotangent`
and fills `input_cotangent`. All arrays must be contiguous and use the primal's
floating-point dtype.

The unit seed is built inside the derivative, as one `select` per argument, so
no caller materialises a tangent vector. `jacfwd_column` takes the index at run
time, which for a wide primal roughly halves the arguments marshalled per call;
it also means every column runs the identical instruction stream and costs the
same, so a warp's lanes can take different columns without diverging.

Prefer `jacfwd_column` on a GPU: `jacfwd`'s matrix costs `n_out * n_args` per
thread, which stops being viable well before the dimensions the column shape
handles comfortably. Because the primal shape differs, neither can share a
build with the scalar-return modes, and neither is implied by the default.

Derivatives are emitted as NVVM LTO IR rather than PTX. Numba-CUDA-MLIR
compiles the calling kernel to LTO IR too whenever a link item is LTO IR, so
nvJitLink inlines the derivative into its caller instead of leaving an opaque
call carrying a parameter per primal argument.

Each public transform builds only its own entry points, and each entry point
carries its own Enzyme marker call. Scalar-output reverse wrappers return an
input-gradient tuple and emit one C-ABI entry point per input argument;
tuple-returning `jacrev` performs one generated reverse sweep per output row at
run time.

CUDA derivatives specialize lazily from the concrete types at each compiled
call site, just like ordinary Numba-CUDA-MLIR device functions, including wide
primals whose calls pass more than 30 arguments. An explicit signature may
optionally constrain the accepted specialization. A compute
capability may also be selected explicitly; otherwise the current device's
compute capability is used:

```python
from numba_cuda_mlir import types

df = grad(
    f,
    signature=types.float64(types.float64, types.float64),
    cc=(8, 0),
)
```

CUDA support currently covers `float32` and `float64` device functions with
homogeneous scalar inputs, returning one scalar or a homogeneous tuple of them.
A primal that writes its outputs through an array argument is not supported;
neither are other array inputs, mixed types, activity annotations, general
mutation, or kernel (`device=False`) differentiation. Derivatives must be called from CUDA-compiled code, not the host.
The LLVM 15 Enzyme bridge currently supports compute capabilities 7.x through
9.x; Numba-CUDA-MLIR's LLVM 20 path for Blackwell (10.x+) needs a matching
newer Enzyme toolchain.

### Development

Install the locked Python environment and bootstrap the matching LLVM 15 and
Enzyme binaries once:

```bash
uv sync
uv run python packaging/bootstrap_dev_toolchain.py
```

The toolchain is extracted into the ignored `.dev-toolchain/` directory and is
discovered automatically; no `PATH` or `NUMBA_ENZYME_PLUGIN_PATH` changes are
needed. The bootstrap is idempotent and accepts `--force` to refresh the
toolchain. The development dependency set includes Numba-CUDA-MLIR's CUDA 12
compiler/runtime components, so a host with a compatible NVIDIA driver can run
the complete suite directly:

```bash
uv run pytest
```


### Scope

* It is still not possible to mark the arguments as active or constant.
* CPU arguments must be floating-point scalars; results may be scalar or
  fixed-size homogeneous tuples of scalars. General array inputs and outputs are not yet supported.
* Linear algebra e.g. `np.dot`, `np.linalg.norm` etc are not supported.

## License

Apache License 2.0 with LLVM Exceptions — see [LICENSE](LICENSE) and
[NOTICE](NOTICE).
