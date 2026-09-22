"""
Synthesise the Enzyme driver module for a lowered kernel.

Builds a second LLVM IR module, via :mod:`llvmlite.ir`'s typed builder
API, declaring the Numba kernel with its exact discovered type and
defining the `__enzyme_autodiff` and `__enzyme_fwddiff` reverse- and
forward-mode entries.

See Also
--------
numba_enzyme.lowering.lower : Produces the `LoweredKernel` this module
    consumes.
numba_enzyme.build.build : Links and compiles the module this module
    produces.

Examples
--------
>>> from numba_enzyme.driver import synthesise
>>> from numba_enzyme.lowering import lower
>>> from numba_enzyme.types import Float64
>>> def f(x: Float64) -> Float64:
...     return x * x
>>> synthesise(lower(f)).grad_symbol  # doctest: +SKIP
'grad__ZN...'
"""

from dataclasses import dataclass

from llvmlite import ir

from numba_enzyme.lowering import ArgSpec, LoweredKernel

_EXCINFO_STRUCT = ir.LiteralStructType(
    [
        ir.IntType(8).as_pointer(),
        ir.IntType(32),
        ir.IntType(8).as_pointer(),
        ir.IntType(8).as_pointer(),
        ir.IntType(32),
    ]
)

# TODO: expand for other scalar types
_SCALAR_IR_TYPE = {
    "double": ir.DoubleType(),
    "float": ir.FloatType(),
    "i32": ir.IntType(32),
    "i64": ir.IntType(64),
    "i8": ir.IntType(8),
}


def _ir_type_from_str(type_str: str) -> ir.Type:
    """
    Parse an LLVM textual type.

    Used to parse arg's type in `LoweredKernel.arg_types` into an
    `llvmlite.ir` type. Also handling the trailing pointer markes
    of the base types.

    Parameters
    ----------
    type_str : str
        LLVM ir type as a string e.g. `"double"`,
        `"i8*"`, `"double*"`.

    Returns
    -------
    llvmlite.ir.Type
        Base types in llvmlite.

    Examples
    --------
    >>> from numba_enzyme.driver import _ir_type_from_str
    >>> str(_ir_type_from_str("i8*"))
    'i8*'
    """
    if type_str.endswith("*"):
        return _ir_type_from_str(type_str[:-1]).as_pointer()
    return _SCALAR_IR_TYPE[type_str]


@dataclass(frozen=True)
class SynthesisedDriver:
    """
    The Enzyme driver module built for one `LoweredKernel`.

    Attributes
    ----------
    ir : str
        The LLVM IR text of the driver module, to be linked
        against the kernel's own IR.
    grad_symbol : str
        Name of the reverse-mode entry point, ``grad_<entry_symbol>``.
    jvp_symbol : str
        Name of the forward-mode entry point, ``jvp_<entry_symbol>``.

    See Also
    --------
    synthesise : Builds a `SynthesisedDriver` instance.

    Examples
    --------
    >>> from numba_enzyme.driver import synthesise
    >>> from numba_enzyme.lowering import lower
    >>> from numba_enzyme.types import Float64
    >>> def f(x: Float64) -> Float64:
    ...     return x * x
    >>> synthesise(lower(f)).jvp_symbol  # doctest: +SKIP
    'jvp__ZN...'
    """

    ir: str
    grad_symbol: str
    jvp_symbol: str


def _target_lines(kernel_ir: str) -> tuple[str, str]:
    """
    Extract the target triple and datalayout kernel's IR.

    Parameters
    ----------
    kernel_ir : str
        The kernel's LLVM IR text emitted by
        `numba_enzyme.lowering.lower`.

    Returns
    -------
    triple : str
        The target triple string.
    datalayout : str
        The target datalayout string.

    Examples
    --------
    >>> from numba_enzyme.driver import _target_lines
    >>> from numba_enzyme.lowering import lower
    >>> from numba_enzyme.types import Float64
    >>> def f(x: Float64) -> Float64:
    ...     return x * x
    >>> _target_lines(lower(f).ir)  # doctest: +SKIP
    ('x86_64-unknown-linux-gnu', 'e-m:e-...')
    """
    triple = datalayout = ""
    for line in kernel_ir.splitlines():
        if line.startswith("target triple"):
            triple = line.split('"')[1]
        elif line.startswith("target datalayout"):
            datalayout = line.split('"')[1]
    return triple, datalayout


def synthesise(kernel: LoweredKernel) -> SynthesisedDriver:
    """
    Build the Enzyme driver module for a lowered kernel.

    Parameters
    ----------
    kernel : numba_enzyme.lowering.LoweredKernel
        The validated, compiled kernel to differentiate.

    Returns
    -------
    SynthesisedDriver
        The driver module defining `grad_<entry>` (reverse-mode) and
        `jvp_<entry>` (forward-mode) for `kernel`.

    See Also
    --------
    SynthesisedDriver : The result this function returns.
    numba_enzyme.lowering.lower : Produces the `kernel` this function
        consumes.

    Examples
    --------
    >>> from numba_enzyme.driver import synthesise
    >>> from numba_enzyme.lowering import lower
    >>> from numba_enzyme.types import Float64
    >>> def f(x: Float64) -> Float64:
    ...     return x * x
    >>> synthesise(lower(f)).grad_symbol  # doctest: +SKIP
    'grad__ZN...'
    """
    retptr_str, _, *flat_strs = kernel.arg_types
    ret_scalar_type = _SCALAR_IR_TYPE[retptr_str[:-1]]
    flat_types = [_ir_type_from_str(s) for s in flat_strs]

    module = ir.Module(name="numba_enzyme_driver")
    module.triple, module.data_layout = _target_lines(kernel.ir)

    excinfo_ptr_type = _EXCINFO_STRUCT.as_pointer()
    kernel_func_type = ir.FunctionType(
        ir.IntType(32),
        [
            ret_scalar_type.as_pointer(),
            excinfo_ptr_type.as_pointer(),
            *flat_types,
        ],
    )
    kernel_fn = ir.Function(module, kernel_func_type, name=kernel.entry_symbol)

    enzyme_dup = ir.GlobalVariable(module, ir.IntType(32), name="enzyme_dup")
    enzyme_dup.linkage = "external"
    enzyme_const = ir.GlobalVariable(module, ir.IntType(32), name="enzyme_const")
    enzyme_const.linkage = "external"

    i8p = ir.IntType(8).as_pointer()
    i64 = ir.IntType(64)
    null_i8p = ir.Constant(i8p, None)
    grad_symbol = f"grad_{kernel.entry_symbol}"
    jvp_symbol = f"jvp_{kernel.entry_symbol}"

    n_scalar_args = sum(1 for spec in kernel.arg_specs if spec.kind == "scalar")

    def _array_param_group(prefix: str, spec: ArgSpec):
        """
        Flattened args used by Enzyme reverse and forward mode.

        For a given array extract
            - data,
            - d_data shadow,
            - nitems,
            - itemsize,
            - shape[0..ndim-1],
            - strides[0..ndim-1].
        meminfo/parent are deliberately NOT parameters here. Assumed to be null
        for the time being as we are not handling the NRT calls.

        Parameters
        ----------
        prefix : str
            Specifies the  logical argument this group belongs to,
            e.g. `"arg0"` for the first argument.

        spec : ArgSpec
            Argument specification for a paremeter of `array` kind.

        Returns
        -------
        List[Tuple[str, llvimlite.ir.Type]
            List of tuple, where the first element is the
            parameter name and the second argument is the
            LLVM type defined in `llvmlite`.
        """
        elem_type = _SCALAR_IR_TYPE[spec.elem_llvm_type]
        elem_ptr = elem_type.as_pointer()
        params = [
            (f"{prefix}_data", elem_ptr),
            (f"{prefix}_ddata", elem_ptr),
            (f"{prefix}_nitems", i64),
            (f"{prefix}_itemsize", i64),
        ]
        params += [(f"{prefix}_shape{d}", i64) for d in range(spec.ndim)]
        params += [(f"{prefix}_strides{d}", i64) for d in range(spec.ndim)]
        return params

    def _array_call_args(
        b,
        enzyme_dup,
        enzyme_const,
        data_ptr,
        d_data_ptr,
        nitems,
        itemsize,
        shapes,
        strides,
    ):
        """
        Enzyme activity markers for one flattened array argument.

        Only `data` carries a gradient/tangent, hence marked with
        `enzyme_dup`. Other fields are metadata Numba needs to know
        where/how many elements to read, hence `enzyme_const` activity
        marker.

        Parameters
        ----------
        b : llvmlite.ir.IRBuilder
            Builder for the function currently under construction.
        enzyme_dup : llvmlite.ir.GlobalVariable
            Enzyme's "duplicated" activity marker.
        enzyme_const : llvmlite.ir.GlobalVariable
            Enzyme's "constant" activity sentinel.
        data_ptr : llvmlite.ir.Value
            The array's data pointer (primal).
        d_data_ptr : llvmlite.ir.Value
            Shadow pointer of array's data.
        nitems : llvmlite.ir.Value
            The array's total element count.
        itemsize : llvmlite.ir.Value
            The array's element size.
        shapes : List[llvmlite.ir.Value]
            Shapes of array along each ndim.
        strides : List[llvmlite.ir.Value]
            Strides of the array along each ndim.

        Returns
        -------
        List[llvmlite.ir.Value]
            The flattened (marker, value[, shadow]) sequence to
            append to the `__enzyme_autodiff`/`__enzyme_fwddiff`
            call's argument list.
        """
        args = [
            b.load(enzyme_const),
            null_i8p,  # meminfo
            b.load(enzyme_const),
            null_i8p,  # parent
            b.load(enzyme_const),
            nitems,
            b.load(enzyme_const),
            itemsize,
            b.load(enzyme_dup),
            data_ptr,
            d_data_ptr,
        ]
        for s in shapes:
            args += [b.load(enzyme_const), s]
        for s in strides:
            args += [b.load(enzyme_const), s]
        return args

    # ---- reverse mode: grad_<entry> ----
    # Enzyme packs the gradients of multiple implicit-active by-value
    # scalar args into a literal struct internally (a single one gets a
    # bare scalar, zero of them means nothing to pack/return); that
    # struct never crosses an external ABI boundary: the packed-scalar
    # part of grad_<entry> is void and writes through an explicit `out`
    # pointer, uniformly regardless of scalar arity, so we never have to
    # replicate the platform's small-vs-large-aggregate return
    # classification ourselves. (A first draft returned the struct
    # directly from grad_<entry> -- silently wrong for n_args=3, a
    # 24-byte struct exceeding x86-64 SysV's 16-byte register-return
    # threshold, since our hand-built IR never lowered it to the
    # required hidden-pointer/sret convention the way a real C frontend
    # would.) Array arguments never go through `out` at all -- Enzyme
    # accumulates their gradient in place into a caller-owned shadow
    # buffer (see `_array_call_args`), so each array argument instead
    # gets its own dedicated group of parameters below.
    grad_ret_type = (
        ir.VoidType()
        if n_scalar_args == 0
        else ret_scalar_type
        if n_scalar_args == 1
        else ir.LiteralStructType([ret_scalar_type] * n_scalar_args)
    )
    autodiff_fn = ir.Function(
        module,
        ir.FunctionType(grad_ret_type, [i8p], var_arg=True),
        name="__enzyme_autodiff",
    )

    out_ptr_type = ret_scalar_type.as_pointer()
    grad_params = [("out", out_ptr_type)] if n_scalar_args else []
    for i, spec in enumerate(kernel.arg_specs):
        if spec.kind == "scalar":
            grad_params.append((f"x{i}", _SCALAR_IR_TYPE[spec.llvm_type]))
        else:
            grad_params += _array_param_group(f"x{i}", spec)

    grad_fn = ir.Function(
        module,
        ir.FunctionType(ir.VoidType(), [t for _, t in grad_params]),
        name=grad_symbol,
    )
    for arg, (name, _) in zip(grad_fn.args, grad_params):
        arg.name = name

    b = ir.IRBuilder(grad_fn.append_basic_block("entry"))
    result_ptr = b.alloca(ret_scalar_type, name="result")
    b.store(ir.Constant(ret_scalar_type, 0.0), result_ptr)
    d_result_ptr = b.alloca(ret_scalar_type, name="d_result")
    b.store(ir.Constant(ret_scalar_type, 1.0), d_result_ptr)
    excinfo_local = b.alloca(excinfo_ptr_type, name="excinfo")
    b.store(ir.Constant(excinfo_ptr_type, None), excinfo_local)
    kernel_i8p = b.bitcast(kernel_fn, i8p)

    call_args = [
        kernel_i8p,
        b.load(enzyme_dup),
        result_ptr,
        d_result_ptr,
        b.load(enzyme_const),
        excinfo_local,
    ]
    remaining_args = iter(grad_fn.args[1:] if n_scalar_args else grad_fn.args)
    for spec in kernel.arg_specs:
        if spec.kind == "scalar":
            call_args.append(next(remaining_args))  # implicit active, no marker
        else:
            data_ptr = next(remaining_args)
            d_data_ptr = next(remaining_args)
            nitems = next(remaining_args)
            itemsize = next(remaining_args)
            shapes = [next(remaining_args) for _ in range(spec.ndim)]
            strides = [next(remaining_args) for _ in range(spec.ndim)]
            call_args += _array_call_args(
                b,
                enzyme_dup,
                enzyme_const,
                data_ptr,
                d_data_ptr,
                nitems,
                itemsize,
                shapes,
                strides,
            )

    grad_result = b.call(autodiff_fn, call_args)
    if n_scalar_args:
        out_ptr = grad_fn.args[0]
        if n_scalar_args == 1:
            b.store(grad_result, out_ptr)
        else:
            for i in range(n_scalar_args):
                elem = b.extract_value(grad_result, i)
                elem_ptr = b.gep(
                    out_ptr, [ir.Constant(ir.IntType(32), i)], inbounds=True
                )
                b.store(elem, elem_ptr)
    b.ret_void()

    # ---- forward mode: jvp_<entry> ----
    # Every scalar arg is an explicit (primal, tangent) enzyme_dup pair
    # (unlike reverse mode, forward mode has no implicit-active
    # convention to lean on); the JVP always lands in d_result's shadow.
    # Each array arg gets the same dedicated parameter group as in
    # grad_<entry>, except its `d_data` slot is the caller-supplied
    # *tangent* (seed) buffer rather than an output accumulator.
    fwddiff_fn = ir.Function(
        module,
        ir.FunctionType(ret_scalar_type, [i8p], var_arg=True),
        name="__enzyme_fwddiff",
    )

    jvp_params = []
    for i, spec in enumerate(kernel.arg_specs):
        if spec.kind == "scalar":
            t = _SCALAR_IR_TYPE[spec.llvm_type]
            jvp_params.append((f"x{i}", t))
            jvp_params.append((f"dx{i}", t))
        else:
            jvp_params += _array_param_group(f"x{i}", spec)

    jvp_fn = ir.Function(
        module,
        ir.FunctionType(ret_scalar_type, [t for _, t in jvp_params]),
        name=jvp_symbol,
    )
    for arg, (name, _) in zip(jvp_fn.args, jvp_params):
        arg.name = name

    b2 = ir.IRBuilder(jvp_fn.append_basic_block("entry"))
    result_ptr2 = b2.alloca(ret_scalar_type, name="result")
    b2.store(ir.Constant(ret_scalar_type, 0.0), result_ptr2)
    d_result_ptr2 = b2.alloca(ret_scalar_type, name="d_result")
    b2.store(ir.Constant(ret_scalar_type, 0.0), d_result_ptr2)
    excinfo_local2 = b2.alloca(excinfo_ptr_type, name="excinfo")
    b2.store(ir.Constant(excinfo_ptr_type, None), excinfo_local2)
    kernel_i8p2 = b2.bitcast(kernel_fn, i8p)

    call_args2 = [
        kernel_i8p2,
        b2.load(enzyme_dup),
        result_ptr2,
        d_result_ptr2,
        b2.load(enzyme_const),
        excinfo_local2,
    ]
    remaining_args2 = iter(jvp_fn.args)
    for spec in kernel.arg_specs:
        if spec.kind == "scalar":
            x = next(remaining_args2)
            dx = next(remaining_args2)
            call_args2 += [b2.load(enzyme_dup), x, dx]
        else:
            data_ptr = next(remaining_args2)
            d_data_ptr = next(remaining_args2)
            nitems = next(remaining_args2)
            itemsize = next(remaining_args2)
            shapes = [next(remaining_args2) for _ in range(spec.ndim)]
            strides = [next(remaining_args2) for _ in range(spec.ndim)]
            call_args2 += _array_call_args(
                b2,
                enzyme_dup,
                enzyme_const,
                data_ptr,
                d_data_ptr,
                nitems,
                itemsize,
                shapes,
                strides,
            )
    b2.call(fwddiff_fn, call_args2)
    b2.ret(b2.load(d_result_ptr2))

    return SynthesisedDriver(
        ir=str(module), grad_symbol=grad_symbol, jvp_symbol=jvp_symbol
    )
