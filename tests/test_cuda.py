"""Compiler and optional live-device tests for the Numba-CUDA-MLIR backend."""

import re

import numpy as np
import pytest
from numba_cuda_mlir import cuda, types

from numba_enzyme.core import (
    grad,
    jacfwd,
    jacfwd_column,
    jacrev,
    jacrev_row,
    jvp,
    vjp,
)
from numba_enzyme.cuda import (
    CUDAEnzymeError,
    CUDALoweredKernel,
    _device_signatures,
    _normalise_signature,
    _parse_compute_capability,
    _primal_shape,
    _sanitize_for_libnvvm,
    _validate_compute_capability,
    _validate_signature,
    _validate_tuple_signature,
    build_cuda,
    is_cuda_device_function,
    synthesise_cuda,
)
from numba_enzyme.types import Float32, Float64


@cuda.jit(device=True)
def annotated_device(x: Float64, y: Float64) -> Float64:
    return x * y + x * x


@cuda.jit(device=True)
def unannotated_device(x, y):
    return x * y + x * x


def annotated_device_grad(x, y):
    return 2 * x + y, x


def _central_diff(func, xs, h=1e-6):
    xs = list(xs)
    gradient = []
    for index in range(len(xs)):
        plus, minus = list(xs), list(xs)
        plus[index] += h
        minus[index] -= h
        gradient.append((func(*plus) - func(*minus)) / (2 * h))
    return tuple(gradient)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Give every CUDA test a fresh derivative cache, like test_gradients.py."""

    from numba_enzyme.cuda import _LOADED_CUDA_DERIVATIVES

    monkeypatch.setenv("NUMBA_ENZYME_CACHE_DIR", str(tmp_path))
    _LOADED_CUDA_DERIVATIVES.clear()
    yield
    _LOADED_CUDA_DERIVATIVES.clear()


def test_cuda_dispatcher_detection_and_annotation_signature():
    assert is_cuda_device_function(annotated_device)
    assert _normalise_signature(annotated_device) == (
        (types.float64, types.float64),
        types.float64,
    )


def test_explicit_signature_overrides_annotations():
    assert _normalise_signature(
        annotated_device, types.float32(types.float32, types.float32)
    ) == ((types.float32, types.float32), types.float32)


def test_cuda_signature_validation_is_deliberately_scalar_and_homogeneous():
    _validate_signature((types.float32,), types.float32)
    with pytest.raises(CUDAEnzymeError, match="same floating-point type"):
        _validate_signature((types.float32,), types.float64)
    with pytest.raises(CUDAEnzymeError, match="only scalar"):
        _validate_signature((types.int32,), types.int32)


@pytest.mark.parametrize(
    "text, expected", [("8.0", (8, 0)), ("80", (8, 0)), ("sm_90", (9, 0))]
)
def test_compute_capability_parser(text, expected):
    assert _parse_compute_capability(text) == expected


def test_compute_capability_matches_llvm_15_bridge():
    assert _validate_compute_capability((8, 9)) == (8, 9)
    with pytest.raises(CUDAEnzymeError, match=r"7.0\+"):
        _validate_compute_capability((6, 1))
    with pytest.raises(CUDAEnzymeError, match="Blackwell"):
        _validate_compute_capability((10, 0))


def test_generated_driver_uses_scalar_c_abi_entry_points():
    kernel = CUDALoweredKernel(
        ir=(
            'target triple = "nvptx64-nvidia-cuda"\n'
            'target datalayout = "e-p:64:64:64-i64:64-n16:32:64"\n'
        ),
        entry_symbol="primal",
        arg_types=(types.float64, types.float64),
        return_type=types.float64,
        compute_capability=(8, 0),
        fastmath=False,
    )
    driver = synthesise_cuda(kernel, "test")

    assert (
        'define double @"numba_enzyme_grad_test_0"('
        'double %"x0", double %"x1")' in driver.ir
    )
    assert (
        'define double @"numba_enzyme_grad_test_1"('
        'double %"x0", double %"x1")' in driver.ir
    )
    assert (
        'define double @"numba_enzyme_jvp_test"('
        'double %"x0", double %"x1", double %"dx0", double %"dx1")' in driver.ir
    )
    assert "__enzyme_autodiff" in driver.ir
    assert "__enzyme_fwddiff" in driver.ir


def test_device_signatures_match_public_grad_and_jvp_shapes():
    signatures = _device_signatures(
        (types.float64, types.float64), types.float64, "scalar"
    )
    # The jacfwd shapes need a tuple-returning primal, so a scalar one has none.
    assert set(signatures) == {"grad", "jvp", "vjp", "jacrev", "jacrev_row"}
    grad_sig = signatures["grad"]
    jvp_sig = signatures["jvp"]
    vjp_sig = signatures["vjp"]
    jacrev_sig = signatures["jacrev"]
    row_sig = signatures["jacrev_row"]
    assert grad_sig.return_type == types.UniTuple(types.float64, 2)
    assert grad_sig.args == (types.float64, types.float64)
    assert jvp_sig.args == (
        types.UniTuple(types.float64, 2),
        types.UniTuple(types.float64, 2),
    )
    assert jvp_sig.return_type == types.float64
    assert vjp_sig.args == (
        types.UniTuple(types.float64, 2),
        types.float64,
    )
    assert vjp_sig.return_type == types.UniTuple(types.float64, 2)
    assert jacrev_sig == grad_sig
    assert row_sig.args == (*grad_sig.args, types.int32)


def test_nvvm_sanitizer_rewrites_enzyme_math_and_new_attributes():
    llvm_ir = """
declare double @llvm.cos.f64(double) #1
define double @f(double %x) #0 {
  %y = call double @llvm.cos.f64(double %x)
  ret double %y
}
attributes #0 = { mustprogress willreturn }
attributes #1 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
"""
    sanitized = _sanitize_for_libnvvm(llvm_ir)
    assert "llvm.cos.f64" not in sanitized
    assert "@__nv_cos" in sanitized
    for attribute in ("mustprogress", "willreturn", "nocallback", "nofree", "nosync"):
        assert attribute not in sanitized
    assert "attributes #0 = { nounwind }" in sanitized


def test_nvvm_sanitizer_rewrites_llvm_8_floating_point_spellings():
    """libNVVM's text reader predates what Enzyme emits for LLVM 15."""

    sanitized = _sanitize_for_libnvvm(
        "  %i12 = fneg fast double %i11\n"
        "  %i78 = select fast i1 %i74, double 0.000000e+00, double %i77\n"
        "  %i90 = phi nnan ninf double [ %i12, %a ], [ %i78, %b ]\n"
    )
    assert "fneg" not in sanitized
    assert "%i12 = fsub fast double -0.000000e+00, %i11" in sanitized
    assert "%i78 = select i1 %i74, double 0.000000e+00, double %i77" in sanitized
    assert "%i90 = phi double [ %i12, %a ], [ %i78, %b ]" in sanitized


def test_core_returns_lazy_cuda_derivatives_without_compiling(monkeypatch):
    def fail_if_eager(*args, **kwargs):
        raise AssertionError("CUDA derivative was built eagerly")

    monkeypatch.setattr("numba_enzyme.cuda.differentiate_cuda", fail_if_eager)
    df = grad(unannotated_device, cc=(8, 0))
    jf = jvp(unannotated_device, cc=(8, 0))
    vf = vjp(unannotated_device, cc=(8, 0))
    jr = jacrev(unannotated_device, cc=(8, 0))
    row = jacrev_row(unannotated_device, cc=(8, 0))

    assert df._numba_enzyme_primal is unannotated_device
    assert df._numba_enzyme_mode == "grad"
    assert jf._numba_enzyme_primal is unannotated_device
    assert jf._numba_enzyme_mode == "jvp"
    assert vf._numba_enzyme_mode == "vjp"
    assert jr._numba_enzyme_mode == "jacrev"
    assert row._numba_enzyme_mode == "jacrev_row"

    # Tuple-returning primals need no signature either; one only constrains.
    tuple_signature = types.UniTuple(types.float64, 2)(types.float64, types.float64)
    for builder in (jacfwd, jacfwd_column, vjp, jacrev, jacrev_row):
        for options in ({}, {"signature": tuple_signature}):
            lazy = builder(tuple_device, cc=(8, 0), **options)
            assert lazy._numba_enzyme_primal is tuple_device
            assert lazy._numba_enzyme_mode == builder.__name__


@pytest.mark.parametrize(
    "builder", [grad, jvp, jacfwd, jacfwd_column, vjp, jacrev, jacrev_row]
)
def test_cpu_functions_reject_cuda_only_options(builder):
    def cpu_function(x: Float32) -> Float32:
        return x * x

    with pytest.raises(TypeError, match="only valid for CUDA"):
        builder(cpu_function, cc=(8, 0))


def _require_numba_cuda_mlir_compiler():
    """Skip only when the optional CUDA compiler itself is unavailable."""

    pytest.importorskip("numba_cuda_mlir")


def test_offline_cuda_pipeline_emits_linkable_lto_ir():
    """Exercise MLIR -> LLVM -> Enzyme -> libNVVM without launching a kernel."""

    _require_numba_cuda_mlir_compiler()
    built = build_cuda(annotated_device, cc=(8, 0))
    df = grad(annotated_device, cc=(8, 0))
    jf = jvp(annotated_device, cc=(8, 0))

    # NVVM's LTO IR wrapper magic. The artifact is opaque bitcode, so the
    # symbols inside it are checked by linking and running, not by inspection.
    assert built.code[:4] == b"\xedCN\x7f"
    assert built.path.suffix == ".ltoir"
    assert df._numba_enzyme_mode == "grad"
    assert jf._numba_enzyme_mode == "jvp"


def test_scalar_reverse_apis_type_inside_device_code_offline():
    """Exercise lazy reverse overload resolution without requiring a GPU."""

    from numba_enzyme.cuda import lower_cuda

    product = vjp(unannotated_device, cc=(8, 0))
    whole = jacrev(unannotated_device, cc=(8, 0))
    row = jacrev_row(unannotated_device, cc=(8, 0))
    namespace = {"product": product, "whole": whole, "row": row}
    exec(  # noqa: S102 - fixed test source
        compile(
            "def use(x, y):\n"
            "    a, b = product((x, y), 2.0)\n"
            "    c, d = whole(x, y)\n"
            "    e, g = row(x, y, 0)\n"
            "    return a + b + c + d + e + g\n",
            "<numba-enzyme-reverse-caller>",
            "exec",
        ),
        namespace,
    )
    caller = cuda.jit(device=True)(namespace["use"])
    lowered = lower_cuda(
        caller,
        signature=types.float64(types.float64, types.float64),
        cc=(8, 0),
    )
    assert lowered.entry_symbol.endswith("usedd")


def test_cuda_grad_and_jvp_execute_on_device():
    """Compare GPU derivatives with analytics and finite differences."""

    if not cuda.is_available():
        pytest.skip("a CUDA GPU is not available")
    _require_numba_cuda_mlir_compiler()

    cc = cuda.get_current_device().compute_capability
    df = grad(unannotated_device, cc=cc)
    jf = jvp(unannotated_device, cc=cc)

    @cuda.jit
    def kernel(x, y, tx, ty, gradient, directional):
        i = cuda.grid(1)
        if i < x.size:
            dx, dy = df(x[i], y[i])
            gradient[i, 0] = dx
            gradient[i, 1] = dy
            directional[i] = jf((x[i], y[i]), (tx[i], ty[i]))

    x = np.asarray([1.0, 2.0, 3.0])
    y = np.asarray([0.5, 1.5, 2.5])
    tx = np.asarray([1.0, -0.5, 0.25])
    ty = np.asarray([0.25, 1.0, -2.0])
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_tx = cuda.to_device(tx)
    d_ty = cuda.to_device(ty)
    d_gradient = cuda.device_array((len(x), 2), dtype=np.float64)
    d_directional = cuda.device_array(len(x), dtype=np.float64)
    kernel[1, 32](d_x, d_y, d_tx, d_ty, d_gradient, d_directional)
    gradient = d_gradient.copy_to_host()
    directional = d_directional.copy_to_host()

    analytic = np.asarray(
        [annotated_device_grad(xi, yi) for xi, yi in zip(x, y, strict=True)]
    )
    finite_difference = np.asarray(
        [
            _central_diff(annotated_device.py_func, (xi, yi))
            for xi, yi in zip(x, y, strict=True)
        ]
    )
    expected_jvp = analytic[:, 0] * tx + analytic[:, 1] * ty

    np.testing.assert_allclose(gradient, analytic, atol=1e-12)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(directional, expected_jvp, atol=1e-12)


def test_scalar_cuda_reverse_apis_execute_on_device():
    """Scalar VJP and Jacobian APIs retain their tuple-oriented interface."""

    if not cuda.is_available():
        pytest.skip("a CUDA GPU is not available")
    _require_numba_cuda_mlir_compiler()

    cc = cuda.get_current_device().compute_capability
    product = vjp(unannotated_device, cc=cc)
    whole = jacrev(unannotated_device, cc=cc)
    one = jacrev_row(unannotated_device, cc=cc)

    @cuda.jit
    def kernel(x, y, cotangent, products, jacobians, rows):
        i = cuda.grid(1)
        if i < x.size:
            products[i, 0], products[i, 1] = product((x[i], y[i]), cotangent[i])
            jacobians[i, 0], jacobians[i, 1] = whole(x[i], y[i])
            rows[i, 0], rows[i, 1] = one(x[i], y[i], 0)

    x = np.asarray([1.0, 2.0, 3.0])
    y = np.asarray([0.5, 1.5, 2.5])
    cotangent = np.asarray([2.0, -1.0, 0.25])
    outputs = [cuda.device_array((len(x), 2), dtype=np.float64) for _ in range(3)]
    kernel[1, 32](
        cuda.to_device(x),
        cuda.to_device(y),
        cuda.to_device(cotangent),
        *outputs,
    )

    expected = np.stack([2 * x + y, x], axis=1)
    np.testing.assert_allclose(outputs[0].copy_to_host(), expected * cotangent[:, None])
    np.testing.assert_allclose(outputs[1].copy_to_host(), expected)
    np.testing.assert_allclose(outputs[2].copy_to_host(), expected)


def test_lazy_cuda_grad_specializes_for_each_call_site_type():
    """Compile float32 and float64 derivatives from one unannotated callable."""

    if not cuda.is_available():
        pytest.skip("a CUDA GPU is not available")

    cc = cuda.get_current_device().compute_capability
    df = grad(unannotated_device, cc=cc)

    @cuda.jit
    def kernel(x32, y32, out32, x64, y64, out64):
        out32[0], out32[1] = df(x32[0], y32[0])
        out64[0], out64[1] = df(x64[0], y64[0])

    x32 = cuda.to_device(np.asarray([2.0], dtype=np.float32))
    y32 = cuda.to_device(np.asarray([1.5], dtype=np.float32))
    x64 = cuda.to_device(np.asarray([3.0], dtype=np.float64))
    y64 = cuda.to_device(np.asarray([2.5], dtype=np.float64))
    out32 = cuda.device_array(2, dtype=np.float32)
    out64 = cuda.device_array(2, dtype=np.float64)
    kernel[1, 1](x32, y32, out32, x64, y64, out64)

    np.testing.assert_allclose(out32.copy_to_host(), [5.5, 2.0])
    np.testing.assert_allclose(out64.copy_to_host(), [8.5, 3.0])

    from numba_enzyme.cuda import _LOADED_CUDA_DERIVATIVES

    assert len(_LOADED_CUDA_DERIVATIVES) == 2


def test_synthesis_emits_only_the_requested_modes():
    """Every emitted entry point costs an Enzyme differentiation of its own."""

    kernel = CUDALoweredKernel(
        ir=(
            'target triple = "nvptx64-nvidia-cuda"\n'
            'target datalayout = "e-p:64:64:64-i64:64-n16:32:64"\n'
        ),
        entry_symbol="primal",
        arg_types=(types.float64, types.float64),
        return_type=types.float64,
        compute_capability=(8, 0),
        fastmath=False,
    )

    only_jvp = synthesise_cuda(kernel, "test", modes=("jvp",))
    assert only_jvp.symbols == {"jvp": ("numba_enzyme_jvp_test",)}
    # Forward machinery only -- no reverse differentiation is requested.
    assert "__enzyme_autodiff" not in only_jvp.ir
    assert only_jvp.ir.count("__enzyme_fwddiff") == 2  # declare + one call

    everything = synthesise_cuda(kernel, "test")
    assert everything.ir.count("__enzyme_autodiff") > 0

    with pytest.raises(CUDAEnzymeError, match="unknown derivative mode"):
        synthesise_cuda(kernel, "test", modes=("nonsense",))
    with pytest.raises(CUDAEnzymeError, match="at least one"):
        synthesise_cuda(kernel, "test", modes=())


def _closure_component(inner):
    """A device function whose source text says nothing about what it calls."""

    namespace = {"_inner": inner}
    exec(  # noqa: S102 - fixed source, only the closed-over callee varies
        compile("def component(x, y):\n    return _inner(x, y)", "<c>", "exec"),
        namespace,
    )
    return cuda.jit(device=True)(namespace["component"])


def test_derivative_cache_sees_through_a_closure():
    """Same source, same name, same signature -- different callee."""

    _require_numba_cuda_mlir_compiler()

    @cuda.jit(device=True)
    def times(x, y):
        return x * y

    @cuda.jit(device=True)
    def plus(x, y):
        return x + y

    signature = types.float64(types.float64, types.float64)
    first = build_cuda(_closure_component(times), signature=signature, cc=(8, 0))
    second = build_cuda(_closure_component(plus), signature=signature, cc=(8, 0))

    assert first.path != second.path
    assert first.symbols["jvp"] != second.symbols["jvp"]


def test_internalise_primal_hides_it_from_the_linker():
    """Two derivatives of same-shaped primals must not define one symbol."""

    from numba_enzyme.cuda import _internalise_primal

    linked = (
        'define double @"_Z1fdd"(double %"x", double %"y") #0 {\n'
        "entry:\n"
        '  ret double %"x"\n'
        "}\n"
        'define double @"numba_enzyme_jvp_k"(double %"x") {\n'
        '  ret double %"x"\n'
        "}\n"
    )
    internalised = _internalise_primal(linked, "_Z1fdd")

    assert 'define internal double @"_Z1fdd"' in internalised
    # Only the primal; the entry point stays externally visible.
    assert 'define double @"numba_enzyme_jvp_k"' in internalised

    with pytest.raises(CUDAEnzymeError, match="exactly one definition"):
        _internalise_primal(linked, "_Z1gdd")


@cuda.jit(device=True)
def tuple_device(x, y):
    return (x * y, x * x + y)


@cuda.jit(device=True)
def single_output_device(x, y):
    return (x * y,)


def _tuple_jacobian(x, y):
    """The analytic Jacobian of `tuple_device`, as (outputs, inputs)."""
    return np.asarray([[y, x], [2 * x, 1.0]])


def test_tuple_and_scalar_modes_cannot_share_a_build():
    with pytest.raises(CUDAEnzymeError, match="scalar-return primal"):
        build_cuda(annotated_device, cc=(8, 0), modes=("grad", "jacfwd"))


def test_forward_jacobian_modes_are_never_implied_by_the_default():
    """The default primal shape is scalar-return, so jacfwd is opt-in."""

    from numba_enzyme.cuda import _normalise_modes

    assert _normalise_modes(None) == {"grad", "jvp"}
    assert _normalise_modes(("jacfwd",)) == frozenset({"jacfwd"})


def test_tuple_reverse_signatures_and_driver_shapes():
    signature = (types.float64, types.float64)
    return_type = types.UniTuple(types.float64, 2)
    signatures = _device_signatures(signature, return_type, "tuple")
    # grad needs a scalar-return primal. jvp and jvp2 are directional, so their
    # result is the whole tangent and a tuple-returning primal supports them,
    # through the array call shape the other tuple modes use.
    assert set(signatures) == {
        "jacfwd",
        "jacfwd_column",
        "jvp",
        "vjp",
        "jacrev",
        "jacrev_row",
    }
    # (tangent, x0, x1, d0, d1): the direction set mirrors the primal's own
    # arguments, one set per sweep.
    assert signatures["jvp"].args == (types.float64[::1],) + (types.float64,) * 4
    assert signatures["vjp"].args == (
        types.float64[::1],
        types.float64[::1],
        types.float64,
        types.float64,
    )
    assert signatures["jacrev"].args == (
        types.float64[:, ::1],
        types.float64,
        types.float64,
    )
    assert signatures["jacrev_row"].args[-1] == types.int32

    kernel = CUDALoweredKernel(
        ir=(
            'target triple = "nvptx64-nvidia-cuda"\n'
            'target datalayout = "e-p:64:64:64-i64:64-n16:32:64"\n'
        ),
        entry_symbol="primal",
        arg_types=signature,
        return_type=return_type,
        compute_capability=(8, 0),
        fastmath=False,
    )
    driver = synthesise_cuda(kernel, "test", modes=("vjp", "jacrev", "jacrev_row"))
    assert driver.symbols == {
        "vjp": ("numba_enzyme_vjp_test",),
        "jacrev": ("numba_enzyme_jacrev_test",),
        "jacrev_row": ("numba_enzyme_jacrevrow_test",),
    }
    assert "__enzyme_fwddiff" not in driver.ir
    assert driver.ir.count("__enzyme_autodiff") == 4  # declaration + three calls
    # The reverse sweeps differentiate the internal scalarisation, not the
    # primal, and the gradient is the only buffer that crosses jacrev_row.
    assert 'define internal double @"numba_enzyme_scalarised_test"' in driver.ir
    assert (
        'define void @"numba_enzyme_jacrevrow_test"(i8* %"gradient_allocated"'
        in driver.ir
    )


def test_offline_tuple_reverse_pipeline_emits_linkable_lto_ir():
    """Run all tuple reverse markers through Enzyme and libNVVM."""

    _require_numba_cuda_mlir_compiler()
    signature = types.UniTuple(types.float64, 2)(types.float64, types.float64)
    built = build_cuda(
        tuple_device,
        signature=signature,
        cc=(8, 0),
        modes=("vjp", "jacrev", "jacrev_row"),
    )
    assert built.code[:4] == b"\xedCN\x7f"
    assert built.shape == "tuple"
    assert built.symbols["vjp"][0].startswith("numba_enzyme_vjp_")
    assert built.symbols["jacrev"][0].startswith("numba_enzyme_jacrev_")
    assert built.symbols["jacrev_row"][0].startswith("numba_enzyme_jacrevrow_")


def test_offline_tuple_pipeline_types_inside_device_code():
    """Build the forward tuple markers, then type every call shape lazily."""

    _require_numba_cuda_mlir_compiler()
    from numba_enzyme.cuda import lower_cuda

    signature = types.UniTuple(types.float64, 2)(types.float64, types.float64)
    built = build_cuda(
        tuple_device, signature=signature, cc=(8, 0), modes=("jacfwd", "jacfwd_column")
    )
    assert built.code[:4] == b"\xedCN\x7f"
    assert built.symbols["jacfwd"][0].startswith("numba_enzyme_jacfwd_")
    assert built.symbols["jacfwd_column"][0].startswith("numba_enzyme_jacfwdcol_")

    whole = jacfwd(tuple_device, cc=(8, 0))
    column = jacfwd_column(tuple_device, cc=(8, 0))
    product = vjp(tuple_device, cc=(8, 0))
    reverse = jacrev(tuple_device, cc=(8, 0))
    row = jacrev_row(tuple_device, cc=(8, 0))

    @cuda.jit(device=True)
    def caller(x, y):
        col = cuda.local.array(2, types.float64)
        cotangent = cuda.local.array(2, types.float64)
        gradient = cuda.local.array(2, types.float64)
        jacobian = cuda.local.array((2, 2), types.float64)
        whole(jacobian, x, y)
        column(col, x, y, 1)
        product(cotangent, gradient, x, y)
        reverse(jacobian, x, y)
        row(gradient, x, y, 0)
        return jacobian[0, 0] + col[1] + gradient[0]

    lowered = lower_cuda(
        caller, signature=types.float64(types.float64, types.float64), cc=(8, 0)
    )
    # A nested function is mangled from its qualified name, not its __name__.
    assert lowered.entry_symbol.endswith("6callerEdd")


def test_lazy_cuda_call_shape_errors_name_the_mismatch():
    """A call matching the wrong primal shape, or neither, says which."""

    _require_numba_cuda_mlir_compiler()
    from numba_enzyme.cuda import lower_cuda

    scalar_only = grad(tuple_device, cc=(8, 0))
    either = jacrev(annotated_device, cc=(8, 0))
    scalar_signature = types.float64(types.float64, types.float64)

    @cuda.jit(device=True)
    def wrong_shape(x, y):
        # grad's two-argument shape belongs to a scalar-return primal.
        return scalar_only(x, y)[0]

    @cuda.jit(device=True)
    def wrong_arity(x, y):
        jacobian = cuda.local.array((2, 2), types.float64)
        either(jacobian, x, y)
        return x

    # grad has no tuple shape at all, so there is no arity to suggest.
    with pytest.raises(
        Exception, match="returns UniTuple.*does not match this CUDA grad call"
    ):
        lower_cuda(wrong_shape, signature=scalar_signature, cc=(8, 0))
    # Three arguments is a valid jacrev call for a tuple-returning primal, so
    # the error names this primal's shape and the arity that shape wants.
    with pytest.raises(Exception, match="takes 2 arguments, not 3"):
        lower_cuda(wrong_arity, signature=scalar_signature, cc=(8, 0))


def test_jacfwd_column_driver_takes_an_index_not_a_tangent():
    """The seed is built inside, so a wide primal marshals far fewer arguments."""

    kernel = CUDALoweredKernel(
        ir=(
            'target triple = "nvptx64-nvidia-cuda"\n'
            'target datalayout = "e-p:64:64:64-i64:64-n16:32:64"\n'
        ),
        entry_symbol="primal",
        arg_types=(types.float64, types.float64),
        return_type=types.UniTuple(types.float64, 2),
        compute_capability=(8, 0),
        fastmath=False,
    )
    driver = synthesise_cuda(kernel, "test", modes=("jacfwd_column",))

    assert driver.symbols["jacfwd_column"] == ("numba_enzyme_jacfwdcol_test",)
    # One memref, the primal's scalars, and one index -- no tangent vector, and
    # no output array, since the primal returns its values in a struct.
    assert (
        'define void @"numba_enzyme_jacfwdcol_test"('
        'i8* %"column_allocated", i8* %"column_aligned", i64 %"column_offset", '
        'i64 %"column_size", i64 %"column_stride", '
        'double %"x0", double %"x1", i32 %"column")' in driver.ir
    )
    # One sweep, and the unit seed is a select per argument rather than a branch.
    assert driver.ir.count("__enzyme_fwddiff") == 2  # declare + one call
    assert len(re.findall(r"select\s+i1", driver.ir)) == 2
    assert "br " not in driver.ir


def test_jacfwd_shapes_agree_column_by_column():
    """The two shapes must be the same derivative, differently arranged."""

    if not cuda.is_available():
        pytest.skip("a CUDA GPU is not available")
    _require_numba_cuda_mlir_compiler()

    cc = cuda.get_current_device().compute_capability
    signature = types.UniTuple(types.float64, 2)(types.float64, types.float64)
    # One explicitly constrained and one inferred, so both paths must agree.
    whole = jacfwd(tuple_device, signature=signature, cc=cc)
    column = jacfwd_column(tuple_device, cc=cc)

    @cuda.jit
    def kernel(x, y, from_whole, from_columns):
        col = cuda.local.array(2, types.float64)
        jac = cuda.local.array((2, 2), types.float64)
        whole(jac, x[0], y[0])
        for r in range(2):
            for c in range(2):
                from_whole[r, c] = jac[r, c]
        for c in range(2):
            column(col, x[0], y[0], c)
            for r in range(2):
                from_columns[r, c] = col[r]

    a = cuda.device_array((2, 2), dtype=np.float64)
    b = cuda.device_array((2, 2), dtype=np.float64)
    kernel[1, 1](
        cuda.to_device(np.asarray([1.5])), cuda.to_device(np.asarray([0.75])), a, b
    )
    np.testing.assert_array_equal(a.copy_to_host(), b.copy_to_host())


def _wide_device_function(n, shape):
    """Generate a device function over `n` scalars, too wide to write out."""

    names = ", ".join(f"x{i}" for i in range(n))
    squares = " + ".join(f"x{i} * x{i}" for i in range(n))
    product = " * ".join(f"x{i}" for i in range(n))
    returned = f"({squares}, {product})" if shape == "tuple" else squares
    source = f"def wide({names}):\n    return {returned}\n"
    namespace = {}
    exec(compile(source, "<wide>", "exec"), namespace)  # noqa: S102 - fixed test source
    return cuda.jit(device=True)(namespace["wide"])


def _wide_kernel(source, **names):
    """Compile a generated kernel whose calls pass more than 30 arguments."""

    namespace = {"cuda": cuda, "types": types, **names}
    exec(compile(source, "<wide-kernel>", "exec"), namespace)  # noqa: S102
    return cuda.jit(namespace["kernel"])


def test_wide_scalar_cuda_derivatives_execute_on_device():
    """Calls past 30 arguments compile their implementation out of line."""

    if not cuda.is_available():
        pytest.skip("a CUDA GPU is not available")
    _require_numba_cuda_mlir_compiler()

    n = 32
    primal = _wide_device_function(n, shape="scalar")
    xs = ", ".join(f"x[{i}]" for i in range(n))
    kernel = _wide_kernel(
        "def kernel(x, gradient, row, product):\n"
        f"    g = df({xs})\n"
        f"    r = one({xs}, 0)\n"
        f"    p = vf(({xs}), 2.0)\n"
        f"    for i in range({n}):\n"
        "        gradient[i] = g[i]\n"
        "        row[i] = r[i]\n"
        "        product[i] = p[i]\n",
        df=grad(primal),
        one=jacrev_row(primal),
        vf=vjp(primal),
    )
    x = 1.0 + 0.01 * np.arange(n)
    gradient, row, product = (cuda.device_array(n, dtype=np.float64) for _ in range(3))
    kernel[1, 1](cuda.to_device(x), gradient, row, product)

    np.testing.assert_allclose(gradient.copy_to_host(), 2 * x, rtol=1e-12)
    np.testing.assert_allclose(row.copy_to_host(), 2 * x, rtol=1e-12)
    np.testing.assert_allclose(product.copy_to_host(), 4 * x, rtol=1e-12)


def test_primal_shape_classifies_both_forms():
    assert _primal_shape(types.float64) == "scalar"
    assert _primal_shape(types.UniTuple(types.float64, 3)) == "tuple"


def test_a_primal_that_returns_nothing_is_rejected():
    with pytest.raises(CUDAEnzymeError, match="has to return its outputs"):
        _validate_signature((types.float64[::1], types.float64), types.void)


def test_tuple_signature_validation():
    scalars = (types.float64, types.float64)
    _validate_tuple_signature(scalars, types.UniTuple(types.float64, 2))
    with pytest.raises(CUDAEnzymeError, match="homogeneous tuple"):
        _validate_tuple_signature(scalars, types.float64)
    with pytest.raises(CUDAEnzymeError, match="element type"):
        _validate_tuple_signature(scalars, types.UniTuple(types.float32, 2))
    with pytest.raises(CUDAEnzymeError, match="at least one argument"):
        _validate_tuple_signature((), types.UniTuple(types.float64, 2))


def test_device_signatures_for_a_tuple_primal():
    scalars = (types.float64, types.float64)
    signatures = _device_signatures(
        scalars, types.UniTuple(types.float64, 2), shape="tuple"
    )

    # No output array and no work buffer: only results cross the call.
    assert signatures["jacfwd_column"].args == (
        types.float64[::1],
        types.float64,
        types.float64,
        types.int32,
    )
    assert signatures["jacfwd"].args == (
        types.float64[:, ::1],
        types.float64,
        types.float64,
    )
    assert signatures["vjp"].args == (
        types.float64[::1],
        types.float64[::1],
        types.float64,
        types.float64,
    )
    assert all(signature.return_type == types.void for signature in signatures.values())


def test_scalar_only_modes_reject_a_tuple_primal():
    signature = types.UniTuple(types.float64, 2)(types.float64, types.float64)
    with pytest.raises(CUDAEnzymeError, match="tuple-returning primal"):
        build_cuda(tuple_device, signature=signature, cc=(8, 0), modes=("grad",))


def test_tuple_externals_are_named_from_the_cache_key():
    """The external behind a tuple implementation has a reproducible symbol."""

    _require_numba_cuda_mlir_compiler()
    from numba_enzyme.cuda import differentiate_cuda

    signature = types.UniTuple(types.float64, 2)(types.float64, types.float64)
    built = differentiate_cuda(
        tuple_device, signature=signature, cc=(8, 0), modes=("jacfwd_column",)
    )
    external = built.externals["jacfwd_column"]

    assert external.name.startswith("numba_enzyme_jacfwdcol_")
    again = differentiate_cuda(
        tuple_device, signature=signature, cc=(8, 0), modes=("jacfwd_column",)
    )
    assert again.externals["jacfwd_column"].name == external.name


def test_tuple_forward_apis_execute_on_device():
    if not cuda.is_available():
        pytest.skip("a CUDA GPU is not available")
    _require_numba_cuda_mlir_compiler()

    cc = cuda.get_current_device().compute_capability
    column = jacfwd_column(tuple_device, cc=cc)
    whole = jacfwd(tuple_device, cc=cc)

    @cuda.jit
    def kernel(x, y, by_column, whole_matrix):
        i = cuda.grid(1)
        if i < x.size:
            col = cuda.local.array(2, types.float64)
            jac = cuda.local.array((2, 2), types.float64)
            for c in range(2):
                column(col, x[i], y[i], c)
                by_column[i, 0, c] = col[0]
                by_column[i, 1, c] = col[1]
            whole(jac, x[i], y[i])
            for r in range(2):
                for c in range(2):
                    whole_matrix[i, r, c] = jac[r, c]

    x = np.asarray([1.0, 2.0, 3.0])
    y = np.asarray([0.5, 1.5, 2.5])
    by_column = cuda.device_array((3, 2, 2), dtype=np.float64)
    whole_matrix = cuda.device_array((3, 2, 2), dtype=np.float64)
    kernel[1, 32](cuda.to_device(x), cuda.to_device(y), by_column, whole_matrix)

    expected = np.stack([_tuple_jacobian(x[i], y[i]) for i in range(3)])
    np.testing.assert_allclose(by_column.copy_to_host(), expected, atol=1e-12)
    np.testing.assert_allclose(whole_matrix.copy_to_host(), expected, atol=1e-12)


def test_tuple_reverse_apis_execute_on_device():
    if not cuda.is_available():
        pytest.skip("a CUDA GPU is not available")
    _require_numba_cuda_mlir_compiler()

    cc = cuda.get_current_device().compute_capability
    product = vjp(tuple_device, cc=cc)
    one_row = jacrev_row(tuple_device, cc=cc)
    whole = jacrev(tuple_device, cc=cc)

    @cuda.jit
    def kernel(x, y, cotangents, products, rows, matrix):
        i = cuda.grid(1)
        if i < x.size:
            cotangent = cuda.local.array(2, types.float64)
            gradient = cuda.local.array(2, types.float64)
            jac = cuda.local.array((2, 2), types.float64)
            cotangent[0] = cotangents[i, 0]
            cotangent[1] = cotangents[i, 1]
            product(cotangent, gradient, x[i], y[i])
            products[i, 0] = gradient[0]
            products[i, 1] = gradient[1]
            for r in range(2):
                one_row(gradient, x[i], y[i], r)
                rows[i, r, 0] = gradient[0]
                rows[i, r, 1] = gradient[1]
            whole(jac, x[i], y[i])
            for r in range(2):
                for c in range(2):
                    matrix[i, r, c] = jac[r, c]

    x = np.asarray([1.0, 2.0, 3.0])
    y = np.asarray([0.5, 1.5, 2.5])
    cotangents = np.asarray([[1.0, 0.0], [0.0, 1.0], [2.0, -0.5]])
    products = cuda.device_array((3, 2), dtype=np.float64)
    rows = cuda.device_array((3, 2, 2), dtype=np.float64)
    matrix = cuda.device_array((3, 2, 2), dtype=np.float64)
    kernel[1, 32](
        cuda.to_device(x),
        cuda.to_device(y),
        cuda.to_device(cotangents),
        products,
        rows,
        matrix,
    )

    expected = np.stack([_tuple_jacobian(x[i], y[i]) for i in range(3)])
    np.testing.assert_allclose(
        products.copy_to_host(),
        np.einsum("no,noi->ni", cotangents, expected),
        atol=1e-12,
    )
    np.testing.assert_allclose(rows.copy_to_host(), expected, atol=1e-12)
    np.testing.assert_allclose(matrix.copy_to_host(), expected, atol=1e-12)


def test_tuple_primal_with_one_output():
    """One output lowers to a bare scalar return rather than a struct."""

    if not cuda.is_available():
        pytest.skip("a CUDA GPU is not available")
    _require_numba_cuda_mlir_compiler()

    cc = cuda.get_current_device().compute_capability
    column = jacfwd_column(single_output_device, cc=cc)

    @cuda.jit
    def kernel(x, y, out):
        i = cuda.grid(1)
        if i < x.size:
            col = cuda.local.array(1, types.float64)
            for c in range(2):
                column(col, x[i], y[i], c)
                out[i, c] = col[0]

    x = np.asarray([1.0, 2.0, 3.0])
    y = np.asarray([0.5, 1.5, 2.5])
    out = cuda.device_array((3, 2), dtype=np.float64)
    kernel[1, 32](cuda.to_device(x), cuda.to_device(y), out)
    np.testing.assert_allclose(out.copy_to_host(), np.stack([y, x], axis=1), atol=1e-12)


def test_wide_tuple_cuda_derivatives_execute_on_device():
    """A tuple primal wider than the 30-argument inlining limit."""

    if not cuda.is_available():
        pytest.skip("a CUDA GPU is not available")
    _require_numba_cuda_mlir_compiler()

    n = 32
    primal = _wide_device_function(n, shape="tuple")
    xs = ", ".join(f"x[{i}]" for i in range(n))
    kernel = _wide_kernel(
        "def kernel(x, column, gradient):\n"
        "    col = cuda.local.array(2, types.float64)\n"
        "    cotangent = cuda.local.array(2, types.float64)\n"
        f"    grad = cuda.local.array({n}, types.float64)\n"
        f"    one(col, {xs}, 5)\n"
        "    column[0] = col[0]\n"
        "    column[1] = col[1]\n"
        "    cotangent[0] = 2.0\n"
        "    cotangent[1] = -0.5\n"
        f"    vf(cotangent, grad, {xs})\n"
        f"    for i in range({n}):\n"
        "        gradient[i] = grad[i]\n",
        one=jacfwd_column(primal),
        vf=vjp(primal),
    )
    x = 1.0 + 0.01 * np.arange(n)
    column = cuda.device_array(2, dtype=np.float64)
    gradient = cuda.device_array(n, dtype=np.float64)
    kernel[1, 1](cuda.to_device(x), column, gradient)

    partials = np.asarray([np.prod(np.delete(x, i)) for i in range(n)])
    np.testing.assert_allclose(
        column.copy_to_host(), [2 * x[5], partials[5]], rtol=1e-12
    )
    np.testing.assert_allclose(
        gradient.copy_to_host(), 2.0 * 2 * x - 0.5 * partials, rtol=1e-12
    )


@cuda.jit(device=True)
def tuple_argument_device(ys, t, ps):
    """A modax-shaped callback: tuple state, scalar time, tuple parameters."""
    return (ys[0] * ps[0], ys[1] + t)


_TUPLE_ARGUMENT_SIGNATURE = types.UniTuple(types.float64, 2)(
    types.UniTuple(types.float64, 2), types.float64, types.UniTuple(types.float64, 1)
)


def test_argument_widths_and_flat_count():
    from numba_enzyme.cuda import _argument_widths, _flat_argument_count

    args = _TUPLE_ARGUMENT_SIGNATURE.args
    assert _argument_widths(args) == (2, None, 1)
    assert _flat_argument_count(args) == 4


def test_tuple_arguments_are_valid_and_mirror_the_call_shape():
    args = _TUPLE_ARGUMENT_SIGNATURE.args
    return_type = _TUPLE_ARGUMENT_SIGNATURE.return_type
    _validate_tuple_signature(args, return_type)

    signatures = _device_signatures(args, return_type, "tuple")
    # A tuple argument becomes an array, of any layout since the entry point
    # reads it through its stride; a scalar argument stays put. The buffers
    # the derivative writes stay contiguous.
    assert signatures["jacfwd_column"].args == (
        types.float64[::1],
        types.float64[:],
        types.float64,
        types.float64[:],
        types.int32,
    )
    assert signatures["vjp"].args == (
        types.float64[::1],
        types.float64[::1],
        types.float64[:],
        types.float64,
        types.float64[:],
    )


def test_a_tuple_argument_must_match_the_return_element_type():
    with pytest.raises(CUDAEnzymeError, match="homogeneous tuples of them"):
        _validate_tuple_signature(
            (types.UniTuple(types.float32, 2),), types.UniTuple(types.float64, 2)
        )


def test_tuple_argument_driver_takes_arrays_not_a_scalar_per_element():
    kernel = CUDALoweredKernel(
        ir=(
            'target triple = "nvptx64-nvidia-cuda"\n'
            'target datalayout = "e-p:64:64:64-i64:64-n16:32:64"\n'
        ),
        entry_symbol="primal",
        arg_types=_TUPLE_ARGUMENT_SIGNATURE.args,
        return_type=_TUPLE_ARGUMENT_SIGNATURE.return_type,
        compute_capability=(8, 0),
        fastmath=False,
    )
    driver = synthesise_cuda(kernel, "test", modes=("jacfwd_column",))

    # column memref, ys memref, the loose scalar t, ps memref, then the index.
    assert (
        'define void @"numba_enzyme_jacfwdcol_test"('
        'i8* %"column_allocated", i8* %"column_aligned", i64 %"column_offset", '
        'i64 %"column_size", i64 %"column_stride", '
        'i8* %"x0_allocated", i8* %"x0_aligned", i64 %"x0_offset", '
        'i64 %"x0_size", i64 %"x0_stride", '
        'double %"x1", '
        'i8* %"x2_allocated", i8* %"x2_aligned", i64 %"x2_offset", '
        'i64 %"x2_size", i64 %"x2_stride", i32 %"column")' in driver.ir
    )
    # Still one sweep, and one seed select per flat scalar, not per argument.
    assert driver.ir.count("__enzyme_fwddiff") == 2
    assert len(re.findall(r"select\s+i1", driver.ir)) == 4


def test_tuple_argument_derivative_executes_on_device():
    if not cuda.is_available():
        pytest.skip("a CUDA GPU is not available")
    _require_numba_cuda_mlir_compiler()

    cc = cuda.get_current_device().compute_capability
    column = jacfwd_column(
        tuple_argument_device, signature=_TUPLE_ARGUMENT_SIGNATURE, cc=cc
    )

    @cuda.jit
    def kernel(ys, t, ps, jacobian):
        i = cuda.grid(1)
        if i < ys.shape[0]:
            col = cuda.local.array(2, types.float64)
            # Five arguments whatever the tuple lengths are.
            for c in range(2):
                column(col, ys[i], t, ps[i], c)
                jacobian[i, 0, c] = col[0]
                jacobian[i, 1, c] = col[1]

    ys = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    ps = np.asarray([[0.5], [1.5]])
    t = 0.25
    d_jacobian = cuda.device_array((2, 2, 2), dtype=np.float64)
    kernel[1, 32](cuda.to_device(ys), t, cuda.to_device(ps), d_jacobian)

    # d(y0 * p0) = (p0, 0); d(y1 + t) = (0, 1)
    expected = np.stack([np.asarray([[ps[i, 0], 0.0], [0.0, 1.0]]) for i in range(2)])
    np.testing.assert_allclose(d_jacobian.copy_to_host(), expected, atol=1e-12)


def test_a_tuple_argument_call_must_pass_an_array():
    _require_numba_cuda_mlir_compiler()
    from numba_enzyme.cuda import lower_cuda

    column = jacfwd_column(
        tuple_argument_device, signature=_TUPLE_ARGUMENT_SIGNATURE, cc=(8, 0)
    )

    @cuda.jit(device=True)
    def caller(x, y):
        col = cuda.local.array(2, types.float64)
        ps = cuda.local.array(1, types.float64)
        column(col, x, y, ps, 0)  # x is a scalar where a state array belongs
        return col[0]

    with pytest.raises(Exception, match="wants a contiguous array there"):
        lower_cuda(
            caller, signature=types.float64(types.float64, types.float64), cc=(8, 0)
        )


def test_a_nested_primal_is_found_under_its_qualified_name():
    """Numba-CUDA-MLIR mangles the qualname, which is not a closure's __name__."""

    _require_numba_cuda_mlir_compiler()
    from numba_enzyme.cuda import lower_cuda

    def make():
        def nested(x, y):
            return (x * y, x + y)

        return cuda.jit(device=True)(nested)

    primal = make()
    assert primal.py_func.__name__ != primal.py_func.__qualname__
    signature = types.UniTuple(types.float64, 2)(types.float64, types.float64)
    kernel = lower_cuda(primal, signature=signature, cc=(8, 0))

    assert "_3clocals_3e" in kernel.entry_symbol
    assert f"@{kernel.entry_symbol}(" in kernel.ir


def test_tuple_jvp_is_the_jacobian_applied_to_a_direction():
    """One sweep must give ``J @ d``, matching the column-by-column assembly."""

    if not cuda.is_available():
        pytest.skip("a CUDA GPU is not available")
    _require_numba_cuda_mlir_compiler()

    cc = cuda.get_current_device().compute_capability
    tangent_of = jvp(tuple_device, cc=cc)

    @cuda.jit
    def kernel(x, y, d, out):
        value = cuda.local.array(2, types.float64)
        tangent_of(value, x[0], y[0], d[0], d[1])
        for r in range(2):
            out[r] = value[r]

    out = cuda.device_array(2, dtype=np.float64)
    direction = np.asarray([0.3, -1.7])
    kernel[1, 1](
        cuda.to_device(np.asarray([1.5])),
        cuda.to_device(np.asarray([0.75])),
        cuda.to_device(direction),
        out,
    )
    expected = _tuple_jacobian(1.5, 0.75) @ direction
    np.testing.assert_allclose(out.copy_to_host(), expected, rtol=1e-12)


def test_multiple_directions_and_cotangents():
    """jvp and vjp take several sweeps at once; jacfwd and jacrev are the
    identity-seeded cases of exactly those loops."""

    if not cuda.is_available():
        pytest.skip("a CUDA GPU is not available")
    _require_numba_cuda_mlir_compiler()

    cc = cuda.get_current_device().compute_capability
    one_direction = jvp(tuple_device, cc=cc)
    two_directions = jvp(tuple_device, cc=cc)
    cotangents = vjp(tuple_device, cc=cc)
    forward = jacfwd(tuple_device, cc=cc)
    backward = jacrev(tuple_device, cc=cc)

    @cuda.jit
    def kernel(x, y, dirs, cots, single, several, one_row, rows, fwd, rev):
        value = cuda.local.array(2, types.float64)
        one_direction(value, x[0], y[0], dirs[0, 0], dirs[0, 1])
        for r in range(2):
            single[r] = value[r]
        two_directions(
            several, x[0], y[0], dirs[0, 0], dirs[0, 1], dirs[1, 0], dirs[1, 1]
        )
        cotangents(cots[0], one_row, x[0], y[0])
        cotangents(cots, rows, x[0], y[0])
        forward(fwd, x[0], y[0])
        backward(rev, x[0], y[0])

    dirs = np.asarray([[0.3, -1.7], [2.1, 0.45]])
    cots = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    buffers = [
        cuda.device_array(shape, dtype=np.float64)
        for shape in (2, (2, 2), 2, (2, 2), (2, 2), (2, 2))
    ]
    kernel[1, 1](
        cuda.to_device(np.asarray([1.5])),
        cuda.to_device(np.asarray([0.75])),
        cuda.to_device(dirs),
        cuda.to_device(cots),
        *buffers,
    )
    single, several, one_row, rows, fwd, rev = (b.copy_to_host() for b in buffers)
    jacobian = _tuple_jacobian(1.5, 0.75)
    np.testing.assert_allclose(single, jacobian @ dirs[0], rtol=1e-12)
    np.testing.assert_allclose(several, (jacobian @ dirs.T).T, rtol=1e-12)
    np.testing.assert_allclose(one_row, cots[0] @ jacobian, rtol=1e-12)
    np.testing.assert_allclose(rows, cots @ jacobian, rtol=1e-12)
    # jacfwd is the identity direction set; jacrev the identity cotangent set.
    np.testing.assert_allclose(fwd, jacobian, rtol=1e-12)
    np.testing.assert_allclose(rev, jacobian, rtol=1e-12)
