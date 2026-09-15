"""
Tests for build.py: correctness of the built shared object, and cache
behavior (hit/miss).
"""

import math

import pytest

from numba_enzyme.build import build
from numba_enzyme.runtime import load
from numba_enzyme.types import Float64


def f(x: Float64) -> Float64:
    return math.sin(x) * x + x * x


def g(x: Float64, y: Float64) -> Float64:
    return math.sin(x) * y + x * y * y


def vector(x: Float64, y: Float64) -> tuple[Float64, Float64]:
    return x * y, x + y


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMBA_ENZYME_CACHE_DIR", str(tmp_path))


def test_build_produces_correct_gradient():
    diff = load(build(f))
    (got,) = diff.grad(1.0)
    expected = math.cos(1.0) * 1.0 + math.sin(1.0) + 2 * 1.0
    assert got == pytest.approx(expected, abs=1e-9)


def test_second_build_is_a_cache_hit_with_no_subprocess_calls(monkeypatch):
    first = build(f)
    assert first.from_cache is False

    import numba_enzyme.build as build_mod

    def _boom(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called on a cache hit")

    monkeypatch.setattr(build_mod.subprocess, "run", _boom)

    second = build(f)
    assert second.from_cache is True
    assert second.path == first.path


def test_second_build_result_is_actually_usable():
    """
    Regression test: a cache-hit BuiltKernel's symbol names must match
    what's actually baked into the .so on disk, not be freshly (and
    differently) recomputed. Numba embeds an internal per-qualname
    version counter into the mangled symbol name that increments on
    every nb.cfunc compilation of "the same" function, even with
    identical source -- so recomputing symbol names via lower() on a
    cache hit silently drifts from the names the cached .so actually
    exports. See build()'s meta.json handling.
    """
    build(f)
    second = build(f)
    assert second.from_cache is True

    diff = load(second)
    (got,) = diff.grad(1.0)
    expected = math.cos(1.0) * 1.0 + math.sin(1.0) + 2 * 1.0
    assert got == pytest.approx(expected, abs=1e-9)


def test_different_function_is_a_cache_miss():
    first = build(f)
    second = build(g)
    assert first.path != second.path
    assert second.from_cache is False


def test_vector_output_metadata_survives_a_cache_hit():
    first = build(vector)
    second = build(vector)

    assert first.n_outputs == second.n_outputs == 2
    assert first.return_type == second.return_type == "double"
    assert first.arg_types == second.arg_types == ("double", "double")
    assert second.vjp_symbol == first.vjp_symbol
    assert second.jvp_symbol == first.jvp_symbol
    assert second.from_cache is True
    assert load(second).jacfwd_column(2.0, 3.0, 1) == pytest.approx((2.0, 1.0))
    assert load(second).jacrev_row(2.0, 3.0, 1) == pytest.approx((1.0, 1.0))
