"""
Tests for build.py: correctness of the built native object, cache
behavior (hit/miss), and MCJIT execution engine lifetime.
"""

import gc
import math

import pytest

from numba_enzyme.build import build
from numba_enzyme.core import grad
from numba_enzyme.runtime import load
from numba_enzyme.types import Float64


def f(x: Float64) -> Float64:
    return math.sin(x) * x + x * x


def g(x: Float64, y: Float64) -> Float64:
    return math.sin(x) * y + x * y * y


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMBA_ENZYME_CACHE_DIR", str(tmp_path))


def test_build_produces_correct_gradient():
    diff = load(build(f))
    (got,) = diff.grad(1.0)
    expected = math.cos(1.0) * 1.0 + math.sin(1.0) + 2 * 1.0
    assert got == pytest.approx(expected, abs=1e-9)


def test_second_build_is_a_cache_hit_with_no_recompilation(monkeypatch):
    first = build(f)
    assert first.from_cache is False

    import numba_enzyme.build as build_mod

    def _boom(*args, **kwargs):
        raise AssertionError("recompilation should not occur on a cache hit")

    monkeypatch.setattr(build_mod, "lower", _boom)
    monkeypatch.setattr(build_mod, "_run_enzyme_pass", _boom)

    second = build(f)
    assert second.from_cache is True
    assert second.path == first.path


def test_second_build_result_is_actually_usable():
    """
    Regression test: a cache-hit BuiltKernel's symbol names must match
    what's actually baked into the object file on disk, not be freshly
    (and differently) recomputed. Numba embeds an internal per-qualname
    version counter into the mangled symbol name that increments on
    every nb.cfunc compilation of "the same" function, even with
    identical source -- so recomputing symbol names via lower() on a
    cache hit silently drifts from the names the cached object actually
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


def test_grad_closure_retains_mcjit_engine_across_gc():
    df = grad(f)
    gc.collect()
    (got,) = df(1.0)
    expected = math.cos(1.0) * 1.0 + math.sin(1.0) + 2 * 1.0
    assert got == pytest.approx(expected, abs=1e-9)
