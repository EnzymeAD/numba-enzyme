"""
End-to-end correctness tests for driver.py: build the Enzyme driver for a
lowered kernel (via build.py + runtime.py) and check the resulting
gradients/JVPs against hand-derived analytic derivatives, across several
functions and arities, both AD modes.
"""

import math

import pytest

from numba_enzyme.build import build
from numba_enzyme.runtime import load
from numba_enzyme.types import Float32, Float64


def f1(x: Float64) -> Float64:
    return math.sin(x) * x + x * x


def f1_grad(x):
    return (math.cos(x) * x + math.sin(x) + 2 * x,)


def f2(x: Float64, y: Float64) -> Float64:
    return math.sin(x) * y + x * y * y


def f2_grad(x, y):
    return (math.cos(x) * y + y * y, math.sin(x) + 2 * x * y)


def f3(x: Float64, y: Float64, z: Float64) -> Float64:
    return x * y + y * z + math.exp(z)


def f3_grad(x, y, z):
    return (y, x + z, y + math.exp(z))


def f_vector(x: Float64, y: Float64) -> tuple[Float64, Float64]:
    return x * y, x * x + y


def f_vector32(x: Float32, y: Float32) -> tuple[Float32, Float32]:
    return x * y, x + y


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMBA_ENZYME_CACHE_DIR", str(tmp_path))


@pytest.mark.parametrize(
    "func,analytic_grad", [(f1, f1_grad), (f2, f2_grad), (f3, f3_grad)]
)
def test_grad_matches_analytic(func, analytic_grad):
    diff = load(build(func))
    xs = tuple(1.0 + 0.3 * i for i in range(diff.n_args))
    expected = analytic_grad(*xs)
    got = diff.grad(*xs)
    for g, e in zip(got, expected):
        assert g == pytest.approx(e, abs=1e-9)


@pytest.mark.parametrize(
    "func,analytic_grad", [(f1, f1_grad), (f2, f2_grad), (f3, f3_grad)]
)
def test_jvp_matches_analytic(func, analytic_grad):
    diff = load(build(func))
    n = diff.n_args
    xs = tuple(1.0 + 0.3 * i for i in range(n))
    expected = analytic_grad(*xs)
    for i in range(n):
        seed = tuple(1.0 if k == i else 0.0 for k in range(n))
        got = diff.jvp(xs, seed)
        assert got == pytest.approx(expected[i], abs=1e-9)


def test_vector_output_forward_derivatives():
    diff = load(build(f_vector))
    x, y = 1.3, 0.7

    assert diff.n_outputs == 2
    assert diff.jvp((x, y), (0.25, -0.5)) == pytest.approx(
        (y * 0.25 + x * -0.5, 2 * x * 0.25 - 0.5), abs=1e-9
    )
    jacobian = diff.jacfwd(x, y)
    assert jacobian[0] == pytest.approx((y, x), abs=1e-9)
    assert jacobian[1] == pytest.approx((2 * x, 1.0), abs=1e-9)
    assert diff.jacfwd_column(x, y, 0) == pytest.approx((y, 2 * x), abs=1e-9)
    assert diff.jacfwd_column(x, y, 1) == pytest.approx((x, 1.0), abs=1e-9)


def test_vector_output_rejects_grad():
    diff = load(build(f_vector))

    with pytest.raises(TypeError, match="scalar-output"):
        diff.grad(1.3, 0.7)


def test_float32_vector_output_uses_float32_runtime_abi():
    diff = load(build(f_vector32))
    jacobian = diff.jacfwd(1.25, 0.75)
    assert jacobian[0] == pytest.approx((0.75, 1.25), abs=1e-6)
    assert jacobian[1] == pytest.approx((1.0, 1.0), abs=1e-6)
