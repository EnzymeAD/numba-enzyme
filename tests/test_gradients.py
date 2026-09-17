"""
Phase 1 test hardening: a broader battery of pure-math functions (N=1..4,
covering products, exp, sin, cos, **, and mixes), cross-checked against
both hand-derived analytics and independent central finite differences,
plus a regression guard against the Phase 0 crash class (a kernel call
routed through a mismatched-type bitcast after linking).
"""

import math

import pytest

from numba_enzyme.build import build
from numba_enzyme.runtime import load
from numba_enzyme.types import Float64


def h1(x: Float64) -> Float64:
    return x**3 - 2 * x


def h1_grad(x):
    return (3 * x**2 - 2,)


def h2(x: Float64, y: Float64) -> Float64:
    return math.cos(x) * y - x**2


def h2_grad(x, y):
    return (-math.sin(x) * y - 2 * x, math.cos(x))


def h3(x: Float64, y: Float64, z: Float64) -> Float64:
    return math.exp(x * y) + z**2


def h3_grad(x, y, z):
    e = math.exp(x * y)
    return (y * e, x * e, 2 * z)


def h4(x: Float64, y: Float64, z: Float64, w: Float64) -> Float64:
    return math.sin(x) * math.cos(y) + z * w - w**2


def h4_grad(x, y, z, w):
    return (
        math.cos(x) * math.cos(y),
        -math.sin(x) * math.sin(y),
        w,
        z - 2 * w,
    )


_CASES = [(h1, h1_grad), (h2, h2_grad), (h3, h3_grad), (h4, h4_grad)]


def _central_diff(func, xs, h=1e-6):
    xs = list(xs)
    grads = []
    for i in range(len(xs)):
        plus, minus = list(xs), list(xs)
        plus[i] += h
        minus[i] -= h
        grads.append((func(*plus) - func(*minus)) / (2 * h))
    return tuple(grads)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMBA_ENZYME_CACHE_DIR", str(tmp_path))


@pytest.mark.parametrize("func,analytic_grad", _CASES)
def test_grad_matches_analytic_and_finite_difference(func, analytic_grad):
    diff = load(build(func))
    n = diff.n_args
    xs = tuple(0.6 + 0.25 * i for i in range(n))

    analytic = analytic_grad(*xs)
    got = diff.grad(*xs)
    for g, a in zip(got, analytic):
        assert g == pytest.approx(a, abs=1e-9)

    fd = _central_diff(func, xs)
    for g, f in zip(got, fd):
        assert g == pytest.approx(f, rel=1e-4, abs=1e-6)


@pytest.mark.parametrize("func,analytic_grad", _CASES)
def test_jvp_matches_analytic(func, analytic_grad):
    diff = load(build(func))
    n = diff.n_args
    xs = tuple(0.6 + 0.25 * i for i in range(n))
    analytic = analytic_grad(*xs)
    for i in range(n):
        seed = tuple(1.0 if k == i else 0.0 for k in range(n))
        got = diff.jvp(xs, seed)
        assert got == pytest.approx(analytic[i], abs=1e-9)


@pytest.mark.parametrize("func,analytic_grad", _CASES)
def test_forward_jacobian_matches_analytic(func, analytic_grad):
    diff = load(build(func))
    n = diff.n_args
    xs = tuple(0.6 + 0.25 * i for i in range(n))
    analytic = analytic_grad(*xs)

    assert diff.jacfwd(*xs) == pytest.approx(analytic, abs=1e-9)


@pytest.mark.parametrize("func,analytic_grad", _CASES)
def test_reverse_products_and_jacobian_match_analytic(func, analytic_grad):
    diff = load(build(func))
    n = diff.n_args
    xs = tuple(0.6 + 0.25 * i for i in range(n))
    analytic = analytic_grad(*xs)

    assert diff.vjp(xs, 2.0) == pytest.approx(
        tuple(2 * value for value in analytic), abs=1e-9
    )
    assert diff.jacrev(*xs) == pytest.approx(analytic, abs=1e-9)


@pytest.mark.parametrize("func", [h1, h2, h3, h4])
def test_no_call_through_a_bitcast_after_linking(func):
    """
    Regression guard for the Phase 0 crash class: a kernel declared with
    a type that doesn't exactly match its real definition forces
    llvm-link to synthesise a call through a bitcasted callee (textually
    "call ... bitcast (...)*(...)"), which crashed Enzyme's forward-mode
    generator. driver.py avoids this by construction -- it declares the
    kernel using its exact real type from LoweredKernel -- but check the
    actual post-link IR directly so a future change that reintroduces a
    type mismatch fails loudly here instead of resurfacing that crash.
    """
    built = build(func)
    combined_ir = (built.path.parent / "combined.ll").read_text()
    for line in combined_ir.splitlines():
        if "call" in line:
            assert "bitcast (" not in line, f"call-through-bitcast reintroduced: {line}"
