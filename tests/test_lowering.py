import inspect
import math

import numpy as np
import pytest

from numba_enzyme.lowering import ArgSpec, LoweredKernel, LoweringError, lower
from numba_enzyme.types import Array1D, Array2D, Float64, Int32


def f1(x: Float64) -> Float64:
    return math.sin(x) * x + x * x


def f2(x: Float64, y: Float64) -> Float64:
    return math.sin(x) * y + x * y * y


def f3(x: Float64, y: Float64, z: Float64) -> Float64:
    return x * y + y * z + math.exp(z)


@pytest.mark.parametrize("func", [f1, f2, f3])
def test_lower_finds_valid_entry_point(func):
    n_args = len(inspect.signature(func).parameters)
    result = lower(func)
    assert isinstance(result, LoweredKernel)
    assert result.n_args == n_args
    assert len(result.arg_types) == n_args + 2
    assert result.arg_types[0] == "double*"
    assert result.arg_types[1] == "{ i8*, i32, i8*, i8*, i32 }**"
    assert result.arg_types[2:] == ("double",) * n_args
    assert f"define i32 @{result.entry_symbol}(" in result.ir


def f_mixed(x: Float64, n: Int32) -> Float64:
    return x * float(n)


def test_lower_supports_mixed_scalar_types():
    result = lower(f_mixed)
    assert result.n_args == 2
    assert result.arg_types == (
        "double*",
        "{ i8*, i32, i8*, i8*, i32 }**",
        "double",
        "i32",
    )


def f_array_1d(x: Array1D(Float64)) -> Float64:
    s = 0.0
    for i in range(x.shape[0]):
        s += x[i] * x[i]
    return s


def test_lower_supports_array_argument():
    result = lower(f_array_1d)
    assert result.n_args == 1
    assert result.arg_specs == (ArgSpec(kind="array", ndim=1, elem_llvm_type="double"),)
    # retptr, excinfo, then the flattened array-struct fields: meminfo,
    # parent, nitems, itemsize, data, shape[0], strides[0] (5 + 2*1 = 7).
    assert result.arg_types == (
        "double*",
        "{ i8*, i32, i8*, i8*, i32 }**",
        "i8*",
        "i8*",
        "i64",
        "i64",
        "double*",
        "i64",
        "i64",
    )


def f_array_2d(x: Array2D(Float64)) -> Float64:
    s = 0.0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            s += x[i, j]
    return s


def test_lower_supports_array_argument_ndim2():
    result = lower(f_array_2d)
    assert result.arg_specs == (ArgSpec(kind="array", ndim=2, elem_llvm_type="double"),)
    assert result.arg_types == (
        "double*",
        "{ i8*, i32, i8*, i8*, i32 }**",
        "i8*",
        "i8*",
        "i64",
        "i64",
        "double*",
        "i64",
        "i64",
        "i64",
        "i64",
    )


def f_mixed_scalar_array(a: Float64, x: Array1D(Float64)) -> Float64:
    s = a
    for i in range(x.shape[0]):
        s += x[i]
    return s


def test_lower_supports_mixed_scalar_and_array_arguments():
    result = lower(f_mixed_scalar_array)
    assert result.n_args == 2
    assert result.arg_specs == (
        ArgSpec(kind="scalar", llvm_type="double"),
        ArgSpec(kind="array", ndim=1, elem_llvm_type="double"),
    )
    assert result.arg_types == (
        "double*",
        "{ i8*, i32, i8*, i8*, i32 }**",
        "double",
        "i8*",
        "i8*",
        "i64",
        "i64",
        "double*",
        "i64",
        "i64",
    )


def f_array_allocates(x: Array1D(Float64)) -> Float64:
    y = np.zeros(3)
    for i in range(3):
        y[i] = x[i] * 2.0
    s = 0.0
    for i in range(3):
        s += y[i]
    return s


def test_lower_rejects_kernel_body_that_needs_nrt():
    with pytest.raises(LoweringError, match="NRT"):
        lower(f_array_allocates)


def f_array_reshape(x: Array1D(Float64)) -> Float64:
    y = x.reshape(x.shape[0], 1)
    s = 0.0
    for i in range(y.shape[0]):
        s += y[i, 0] * y[i, 0]
    return s


def test_lower_allows_refcount_only_nrt_calls():
    result = lower(f_array_reshape)
    assert "@NRT_incref" in result.ir
    assert "@NRT_decref" in result.ir
    assert "@NRT_MemInfo_alloc_aligned" not in result.ir


def test_lower_marks_nrt_dtor_nofree_and_inactive():
    result = lower(f_array_reshape)
    dtor_line = next(
        line
        for line in result.ir.splitlines()
        if "declare" in line and "NRT_MemInfo_call_dtor" in line
    )
    assert "nofree" in dtor_line
    assert '"enzyme_inactive"' in dtor_line


def test_lower_still_rejects_nrt_allocation_calls():
    with pytest.raises(LoweringError, match="NRT_MemInfo_alloc_aligned"):
        lower(f_array_allocates)
