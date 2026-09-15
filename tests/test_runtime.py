"""
Tests for runtime.py: the grad/jvp callables it produces behave correctly
and reject wrong-arity calls.
"""

import math

import pytest

from numba_enzyme.build import build
from numba_enzyme.runtime import Differentiable, load
from numba_enzyme.types import Float64


def f(x: Float64, y: Float64) -> Float64:
    return math.sin(x) * y + x * y * y


def f_vector(x: Float64, y: Float64) -> tuple[Float64, Float64]:
    return x * y, x + y


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMBA_ENZYME_CACHE_DIR", str(tmp_path))


def test_load_derivatives_match_analytic():
    diff = load(build(f))
    assert isinstance(diff, Differentiable)
    assert diff.n_args == 2

    x, y = 1.3, 0.7
    got_grad = diff.grad(x, y)
    expected_grad = (math.cos(x) * y + y * y, math.sin(x) + 2 * x * y)
    assert got_grad == pytest.approx(expected_grad, abs=1e-9)

    got_jvp_x = diff.jvp((x, y), (1.0, 0.0))
    got_jvp_y = diff.jvp((x, y), (0.0, 1.0))
    assert got_jvp_x == pytest.approx(expected_grad[0], abs=1e-9)
    assert got_jvp_y == pytest.approx(expected_grad[1], abs=1e-9)
    assert diff.jacfwd(x, y) == pytest.approx(expected_grad, abs=1e-9)
    assert diff.jacfwd_column(x, y, 0) == pytest.approx(expected_grad[0], abs=1e-9)
    assert diff.jacfwd_column(x, y, 1) == pytest.approx(expected_grad[1], abs=1e-9)


def test_grad_rejects_wrong_arity():
    diff = load(build(f))
    with pytest.raises(TypeError):
        diff.grad(1.0)


def test_grad_rejects_vector_output():
    diff = load(build(f_vector))
    with pytest.raises(TypeError, match="requires a scalar-output function"):
        diff.grad(1.0, 2.0)


def test_jvp_rejects_wrong_arity():
    diff = load(build(f))
    with pytest.raises(TypeError):
        diff.jvp((1.0, 2.0), (1.0,))


def test_jacfwd_rejects_wrong_arity():
    diff = load(build(f))
    with pytest.raises(TypeError, match="expected 2 arguments"):
        diff.jacfwd(1.0)


def test_jacfwd_column_validates_arguments_and_index():
    diff = load(build(f))
    with pytest.raises(TypeError, match="2 primal arguments"):
        diff.jacfwd_column(1.0, 0)
    with pytest.raises(TypeError, match="must be an integer"):
        diff.jacfwd_column(1.0, 2.0, 0.5)
    with pytest.raises(IndexError, match="out of range"):
        diff.jacfwd_column(1.0, 2.0, 2)
    with pytest.raises(IndexError, match="out of range"):
        diff.jacfwd_column(1.0, 2.0, -1)
