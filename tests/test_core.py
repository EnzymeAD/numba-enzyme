"""Tests for core.py's public differentiation API."""

import math

import pytest

from numba_enzyme.core import (
    differentiable,
    grad,
    jvp,
)
from numba_enzyme.types import Float64


def f(x: Float64, y: Float64) -> Float64:
    return math.sin(x) * y + x * y * y


def f_grad(x, y):
    return (math.cos(x) * y + y * y, math.sin(x) + 2 * x * y)


def f_vector(x: Float64, y: Float64) -> tuple[Float64, Float64]:
    return x * y, x * x + y


# Decorated at module level -- must stay cheap (no build triggered by
# decoration itself, only by actually accessing .grad/.jvp), otherwise
# just importing this test module would build outside the isolated cache
# the autouse fixture below sets up per-test.
@differentiable
def f_decorated(x: Float64, y: Float64) -> Float64:
    return math.sin(x) * y + x * y * y


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMBA_ENZYME_CACHE_DIR", str(tmp_path))


def test_grad_matches_analytic():
    x, y = 1.3, 0.7
    assert grad(f)(x, y) == pytest.approx(f_grad(x, y), abs=1e-9)


def test_jvp_matches_analytic():
    x, y = 1.3, 0.7
    expected = f_grad(x, y)
    j = jvp(f)
    assert j((x, y), (1.0, 0.0)) == pytest.approx(expected[0], abs=1e-9)
    assert j((x, y), (0.0, 1.0)) == pytest.approx(expected[1], abs=1e-9)


def test_grad_rejects_vector_output():
    with pytest.raises(TypeError, match="grad requires a scalar-output function"):
        grad(f_vector)


def test_differentiable_wrapper_calls_original_function():
    x, y = 1.3, 0.7
    assert f_decorated(x, y) == f(x, y)


def test_differentiable_wrapper_exposes_all_derivative_modes():
    x, y = 1.3, 0.7
    expected = f_grad(x, y)
    assert f_decorated.grad(x, y) == pytest.approx(expected, abs=1e-9)
    assert f_decorated.jvp((x, y), (1.0, 0.0)) == pytest.approx(expected[0], abs=1e-9)


def test_differentiable_grad_is_cached():
    grad_callable = f_decorated.grad
    assert f_decorated.grad is grad_callable
