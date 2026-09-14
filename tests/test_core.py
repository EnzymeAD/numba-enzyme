"""Tests for core.py's public differentiation API."""

import math

import numba as nb
import numpy as np
import pytest

from numba_enzyme.core import (
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


def test_jacfwd_matches_analytic():
    x, y = 1.3, 0.7
    assert jacfwd(f)(x, y) == pytest.approx(f_grad(x, y), abs=1e-9)


def test_jacfwd_column_matches_analytic():
    x, y = 1.3, 0.7
    expected = f_grad(x, y)
    column = jacfwd_column(f)
    assert column(x, y, 0) == pytest.approx(expected[0], abs=1e-9)
    assert column(x, y, 1) == pytest.approx(expected[1], abs=1e-9)


def test_scalar_output_public_reverse_apis():
    x, y = 1.3, 0.7
    expected = f_grad(x, y)
    assert vjp(f)((x, y), 2.0) == pytest.approx(
        tuple(2 * value for value in expected), abs=1e-9
    )
    assert jacrev(f)(x, y) == pytest.approx(expected, abs=1e-9)
    assert jacrev_row(f)(x, y, 0) == pytest.approx(expected, abs=1e-9)


def test_vector_output_public_forward_apis():
    x, y = 1.3, 0.7
    assert jvp(f_vector)((x, y), (1.0, 0.0)) == pytest.approx((y, 2 * x))
    jacobian = jacfwd(f_vector)(x, y)
    assert jacobian[0] == pytest.approx((y, x))
    assert jacobian[1] == pytest.approx((2 * x, 1.0))
    assert jacfwd_column(f_vector)(x, y, 1) == pytest.approx((x, 1.0))


def test_vector_output_public_reverse_apis():
    x, y = 1.3, 0.7
    assert vjp(f_vector)((x, y), (0.25, -0.5)) == pytest.approx(
        (y * 0.25 - x, x * 0.25 - 0.5)
    )
    jacobian = jacrev(f_vector)(x, y)
    assert jacobian[0] == pytest.approx((y, x))
    assert jacobian[1] == pytest.approx((2 * x, 1.0))
    assert jacrev_row(f_vector)(x, y, 1) == pytest.approx((2 * x, 1.0))


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
    assert f_decorated.jacfwd(x, y) == pytest.approx(expected, abs=1e-9)
    assert f_decorated.jacfwd_column(x, y, 1) == pytest.approx(expected[1], abs=1e-9)
    assert f_decorated.vjp((x, y), 1.0) == pytest.approx(expected, abs=1e-9)
    assert f_decorated.jacrev(x, y) == pytest.approx(expected, abs=1e-9)
    assert f_decorated.jacrev_row(x, y, 0) == pytest.approx(expected, abs=1e-9)


def test_differentiable_grad_is_cached():
    grad_callable = f_decorated.grad
    assert f_decorated.grad is grad_callable


def test_differentiable_forward_jacobian_callables_are_cached():
    whole = f_decorated.jacfwd
    column = f_decorated.jacfwd_column
    assert f_decorated.jacfwd is whole
    assert f_decorated.jacfwd_column is column


def test_differentiable_reverse_jacobian_callables_are_cached():
    product = f_decorated.vjp
    whole = f_decorated.jacrev
    row = f_decorated.jacrev_row
    assert f_decorated.vjp is product
    assert f_decorated.jacrev is whole
    assert f_decorated.jacrev_row is row


def f_plain(x, y):
    return math.sin(x) * y + x * y * y


def f_plain_vector(x, y):
    return x * y, x * x + y


def test_unannotated_functions_need_no_types():
    x, y = 1.3, 0.7
    expected = f_grad(x, y)
    assert grad(f_plain)(x, y) == pytest.approx(expected, abs=1e-9)
    assert jvp(f_plain)((x, y), (1.0, 0.0)) == pytest.approx(expected[0], abs=1e-9)
    assert vjp(f_plain)((x, y), 2.0) == pytest.approx(
        tuple(2 * value for value in expected), abs=1e-9
    )
    assert jacfwd(f_plain)(x, y) == pytest.approx(expected, abs=1e-9)
    assert jacfwd_column(f_plain)(x, y, 1) == pytest.approx(expected[1], abs=1e-9)
    assert jacrev(f_plain)(x, y) == pytest.approx(expected, abs=1e-9)
    assert jacrev_row(f_plain)(x, y, 0) == pytest.approx(expected, abs=1e-9)


def test_unannotated_vector_function_infers_its_tuple_result():
    x, y = 1.3, 0.7
    jacobian = jacfwd(f_plain_vector)(x, y)
    assert jacobian[0] == pytest.approx((y, x))
    assert jacobian[1] == pytest.approx((2 * x, 1.0))
    assert jacrev_row(f_plain_vector)(x, y, 1) == pytest.approx((2 * x, 1.0))
    assert vjp(f_plain_vector)((x, y), (0.25, -0.5)) == pytest.approx(
        (y * 0.25 - x, x * 0.25 - 0.5)
    )
    # The result shape is unknown until the first call, so grad raises then.
    vector_grad = grad(f_plain_vector)
    with pytest.raises(TypeError, match="grad requires a scalar-output function"):
        vector_grad(x, y)


def test_unannotated_transform_builds_on_first_call(monkeypatch):
    def cube(x):
        return x * x * x

    def fail(*args, **kwargs):
        raise AssertionError("built before the first call")

    monkeypatch.setattr("numba_enzyme.runtime.build", fail)
    derivative = grad(cube)
    with pytest.raises(AssertionError, match="before the first call"):
        derivative(2.0)


def test_unannotated_function_specializes_per_argument_type():
    from numba_enzyme.runtime import _SPECIALIZATIONS

    def square(x):
        return x * x

    derivative = grad(square)
    assert derivative(3.0) == pytest.approx((6.0,))
    assert derivative(np.float32(1.5)) == pytest.approx((3.0,))
    assert jacrev(square)(4.0) == pytest.approx((8.0,))
    # jacrev reuses grad's float64 build rather than building its own.
    assert {types for func, types in _SPECIALIZATIONS if func is square} == {
        (nb.types.float64,),
        (nb.types.float32,),
    }


def test_unannotated_function_rejects_non_float_arguments():
    def square(x):
        return x * x

    with pytest.raises(TypeError, match="floating-point arguments"):
        grad(square)(3)
    with pytest.raises(TypeError, match="expected 1 primal values"):
        grad(square)(3.0, 4.0)


def test_differentiable_wrapper_accepts_unannotated_functions():
    @differentiable
    def product(x, y):
        return x * y

    assert product(2.0, 3.0) == 6.0
    assert product.grad(2.0, 3.0) == pytest.approx((3.0, 2.0))
    assert product.jacfwd_column(2.0, 3.0, 0) == pytest.approx(3.0)
