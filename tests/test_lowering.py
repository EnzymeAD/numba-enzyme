import inspect
import math

import pytest

from numba_enzyme.lowering import LoweredKernel, LoweringError, lower
from numba_enzyme.types import Float64, Int32


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
    assert result.n_outputs == 1
    assert result.return_type == "double"
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
    assert result.n_outputs == 1
    assert result.arg_types == (
        "double*",
        "{ i8*, i32, i8*, i8*, i32 }**",
        "double",
        "i32",
    )


def f_vector(x: Float64, y: Float64) -> tuple[Float64, Float64]:
    return x * y, x * x + y


def test_lower_supports_fixed_homogeneous_tuple_returns():
    result = lower(f_vector)
    assert result.n_args == 2
    assert result.n_outputs == 2
    assert result.return_type == "double"
    assert result.arg_types == (
        "[2 x double]*",
        "{ i8*, i32, i8*, i8*, i32 }**",
        "double",
        "double",
    )


def f_heterogeneous(x: Float64) -> tuple[Float64, Int32]:
    return x, 1


def f_variadic(x: Float64) -> tuple[Float64, ...]:
    return (x,)


@pytest.mark.parametrize(
    "func, message",
    [
        (f_heterogeneous, "must be homogeneous"),
        (f_variadic, "must have a fixed size"),
    ],
)
def test_lower_rejects_unsupported_tuple_returns(func, message):
    with pytest.raises(LoweringError, match=message):
        lower(func)
