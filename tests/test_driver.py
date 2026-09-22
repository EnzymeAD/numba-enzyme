"""
End-to-end correctness tests for driver.py: build the Enzyme driver for a
lowered kernel (via build.py + runtime.py) and check the resulting
gradients/JVPs against hand-derived analytic derivatives, across several
functions and arities, both AD modes.
"""

import math

import numpy as np
import pytest

from numba_enzyme.build import build
from numba_enzyme.driver import synthesise
from numba_enzyme.lowering import lower
from numba_enzyme.runtime import load
from numba_enzyme.types import Array1D, Float64


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


def farr(x: Array1D(Float64)) -> Float64:
    s = 0.0
    for i in range(x.shape[0]):
        s += x[i] * x[i]
    return s


def farr_grad(x):
    return 2 * x


def test_synthesise_array_grad_signature():
    """
    IR-shape check, before ever running Enzyme: a single array argument
    (no scalar args) means grad_<entry> has no packed `out` pointer at
    all -- there is nothing for __enzyme_autodiff to pack/return, since
    the array's gradient is instead accumulated in place into its own
    dedicated shadow-pointer parameter (x0_ddata).
    """
    drv = synthesise(lower(farr))
    signature = (
        f'define void @"{drv.grad_symbol}"(double* %"x0_data", '
        'double* %"x0_ddata", i64 %"x0_nitems", i64 %"x0_itemsize", '
        'i64 %"x0_shape0", i64 %"x0_strides0")'
    )
    assert signature in drv.ir
    grad_start = drv.ir.index(signature)
    grad_body = drv.ir[grad_start : drv.ir.index("\n}\n", grad_start)]
    assert 'call void (i8*, ...) @"__enzyme_autodiff"' in grad_body
    # 2 enzyme_dup markers (retptr's cotangent seed, and the array's
    # data/d_data pair) + 7 enzyme_const markers (excinfo, meminfo,
    # parent, nitems, itemsize, shape, strides).
    assert grad_body.count('@"enzyme_dup"') == 2
    assert grad_body.count('@"enzyme_const"') == 7


def test_grad_and_jvp_match_analytic_for_array_argument():
    diff = load(build(farr))
    x = np.array([1.0, 2.0, 3.0])
    (g,) = diff.grad(x)
    assert g == pytest.approx(farr_grad(x), abs=1e-9)

    for i in range(len(x)):
        seed = np.zeros_like(x)
        seed[i] = 1.0
        got = diff.jvp((x,), (seed,))
        assert got == pytest.approx(farr_grad(x)[i], abs=1e-9)
