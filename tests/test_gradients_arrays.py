"""
End-to-end reverse-mode/forward-mode tests for numpy.ndarray arguments
(Phase 1/2 of array support), mirroring test_gradients.py's analytic +
central-finite-difference cross-check pattern for the scalar-only path.

Kernel bodies here deliberately use a plain Python indexing loop, not a
numpy ufunc reduction (e.g. np.sum/np.dot) -- the latter isn't yet
supported (see README's Scope section), and a plain loop is already
provably differentiable through Enzyme's core (non-BLAS) activity
analysis.
"""

import numpy as np
import pytest

from numba_enzyme.build import build
from numba_enzyme.runtime import load
from numba_enzyme.types import Array1D, Float64


def sum_of_squares(x: Array1D(Float64)) -> Float64:
    s = 0.0
    for i in range(x.shape[0]):
        s += x[i] * x[i]
    return s


def sum_of_squares_grad(x):
    return 2 * x


def weighted_sum(a: Float64, x: Array1D(Float64)) -> Float64:
    s = a
    for i in range(x.shape[0]):
        s += x[i] * x[i]
    return s


def weighted_sum_grad(a, x):
    return 1.0, 2 * x


_ARRAY_CASES = [sum_of_squares]
_ARRAY_LENGTHS = [3, 8]


def _central_diff_array(func, x, h=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        plus, minus = x.copy(), x.copy()
        plus[i] += h
        minus[i] -= h
        grad[i] = (func(plus) - func(minus)) / (2 * h)
    return grad


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMBA_ENZYME_CACHE_DIR", str(tmp_path))


@pytest.mark.parametrize("func", _ARRAY_CASES)
@pytest.mark.parametrize("n", _ARRAY_LENGTHS)
def test_grad_matches_analytic_and_finite_difference(func, n):
    diff = load(build(func))
    x = np.array([0.6 + 0.25 * i for i in range(n)])

    (got,) = diff.grad(x)
    assert isinstance(got, np.ndarray)
    assert got.shape == x.shape
    np.testing.assert_allclose(got, sum_of_squares_grad(x), atol=1e-9)

    fd = _central_diff_array(func, x)
    np.testing.assert_allclose(got, fd, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("func", _ARRAY_CASES)
@pytest.mark.parametrize("n", _ARRAY_LENGTHS)
def test_jvp_matches_analytic_one_hot_seeds(func, n):
    diff = load(build(func))
    x = np.array([0.6 + 0.25 * i for i in range(n)])
    analytic = sum_of_squares_grad(x)

    for i in range(n):
        seed = np.zeros_like(x)
        seed[i] = 1.0
        got = diff.jvp((x,), (seed,))
        assert got == pytest.approx(analytic[i], abs=1e-9)


def test_jvp_matches_directional_derivative():
    diff = load(build(sum_of_squares))
    x = np.array([1.0, -2.0, 3.5])
    seed = np.array([0.3, 0.7, -1.1])
    got = diff.jvp((x,), (seed,))
    expected = float(sum_of_squares_grad(x) @ seed)
    assert got == pytest.approx(expected, abs=1e-9)


def test_grad_matches_analytic_mixed_scalar_and_array():
    diff = load(build(weighted_sum))
    a = 1.5
    x = np.array([0.6, 0.85, 1.1])
    g_a, g_x = diff.grad(a, x)
    exp_a, exp_x = weighted_sum_grad(a, x)
    assert g_a == pytest.approx(exp_a, abs=1e-9)
    np.testing.assert_allclose(g_x, exp_x, atol=1e-9)


def test_grad_rejects_wrong_argument_kind():
    diff = load(build(sum_of_squares))
    with pytest.raises(TypeError):
        diff.grad(1.0)  # scalar where an array is expected


def test_grad_rejects_non_contiguous_array():
    diff = load(build(sum_of_squares))
    x = np.array([1.0, 2.0, 3.0, 4.0])[::2]  # a non-contiguous view
    assert not x.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError):
        diff.grad(x)
