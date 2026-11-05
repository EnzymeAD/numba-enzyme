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
>>> synthesise(lower(f)).vjp_symbol  # doctest: +SKIP
'vjp__ZN...'
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
_EXCINFO_PTR_TYPE = _EXCINFO_STRUCT.as_pointer()
_I8P = ir.IntType(8).as_pointer()
_I32 = ir.IntType(32)
_I64 = ir.IntType(64)

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
    jvp_symbol : str
        Name of the forward-mode entry point, ``jvp_<entry_symbol>``.
    vjp_symbol : str
        Name of the reverse-mode vector-Jacobian product entry point,
        ``vjp_<entry_symbol>``. A gradient is this product seeded with one,
        so reverse mode needs no second entry point of its own.

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
    jvp_symbol: str
    vjp_symbol: str


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


@dataclass(frozen=True)
class _Primal:
    """
    The declarations every entry point in one driver module shares.

    Attributes
    ----------
    kernel : numba_enzyme.lowering.LoweredKernel
        The lowered primal being differentiated.
    module : llvmlite.ir.Module
        Module the entry points are emitted into.
    function : llvmlite.ir.Function
        Declaration of the Numba primal, with its exact discovered type.
    dup : llvmlite.ir.GlobalVariable
        Enzyme's duplicated-activity marker.
    const : llvmlite.ir.GlobalVariable
        Enzyme's constant-activity marker.
    scalar_type : llvmlite.ir.Type
        LLVM type of one result component.
    storage_type : llvmlite.ir.Type
        LLVM type holding a whole result: the component type itself for a
        scalar primal, an array of it for a tuple-valued one.

    See Also
    --------
    synthesise : Builds the `_Primal` the entry points are emitted against.
    """

    kernel: LoweredKernel
    module: ir.Module
    function: ir.Function
    dup: ir.GlobalVariable
    const: ir.GlobalVariable
    scalar_type: ir.Type
    storage_type: ir.Type


def _array_param_group(prefix: str, spec: ArgSpec) -> list:
    """
    Name and type the entry-point parameters of one array argument.

    For a given array these are its data pointer, the data's shadow, nitems,
    itemsize, then `ndim` shapes and `ndim` strides. meminfo and parent are
    deliberately not parameters: they are passed as null, since NRT
    allocation is not supported.

    Parameters
    ----------
    prefix : str
        Name of the logical argument this group belongs to, e.g. ``"x0"``.
    spec : numba_enzyme.lowering.ArgSpec
        Specification of an argument of `array` kind.

    Returns
    -------
    list of tuple of (str, llvmlite.ir.Type)
        Each parameter's name and LLVM type, in order.
    """
    elem_ptr = _SCALAR_IR_TYPE[spec.elem_llvm_type].as_pointer()
    params = [
        (f"{prefix}_data", elem_ptr),
        (f"{prefix}_ddata", elem_ptr),
        (f"{prefix}_nitems", _I64),
        (f"{prefix}_itemsize", _I64),
    ]
    params += [(f"{prefix}_shape{d}", _I64) for d in range(spec.ndim)]
    params += [(f"{prefix}_strides{d}", _I64) for d in range(spec.ndim)]
    return params


def _argument_params(kernel: LoweredKernel, forward: bool) -> list:
    """
    Name and type the entry-point parameters carrying the primal arguments.

    Parameters
    ----------
    kernel : numba_enzyme.lowering.LoweredKernel
        The lowered primal being differentiated.
    forward : bool
        Whether the entry point is forward mode, which pairs every scalar
        argument with its tangent. Reverse mode passes scalars alone.

    Returns
    -------
    list of tuple of (str, llvmlite.ir.Type)
        Each parameter's name and LLVM type, in order.

    See Also
    --------
    _marker_args : Consumes these parameters in the same order.
    """
    params = []
    for index, spec in enumerate(kernel.arg_specs):
        if spec.kind == "scalar":
            arg_type = _SCALAR_IR_TYPE[spec.llvm_type]
            params.append((f"x{index}", arg_type))
            if forward:
                params.append((f"dx{index}", arg_type))
        else:
            params += _array_param_group(f"x{index}", spec)
    return params


def _marker_args(primal: _Primal, builder, args, forward: bool) -> list:
    """
    Build the activity-marked marker-call arguments for the primal arguments.

    In reverse mode a scalar argument is implicitly active and passed bare;
    in forward mode it is a duplicated ``(primal, tangent)`` pair. For an
    array, only the data pointer carries a derivative, so it is duplicated
    with its shadow while the remaining fields are constant.

    Parameters
    ----------
    primal : _Primal
        Shared declarations of the driver module being built.
    builder : llvmlite.ir.IRBuilder
        Builder positioned in the emitting entry point's block.
    args : iterator of llvmlite.ir.Value
        The entry point's argument parameters, laid out as
        `_argument_params` names them.
    forward : bool
        Whether the marker is Enzyme's forward-mode one.

    Returns
    -------
    list of llvmlite.ir.Value
        Arguments to append to the marker call.
    """
    null_i8p = ir.Constant(_I8P, None)
    call_args = []
    for spec in primal.kernel.arg_specs:
        if spec.kind == "scalar":
            if forward:
                call_args += [builder.load(primal.dup), next(args), next(args)]
            else:
                call_args.append(next(args))
            continue
        data, d_data, nitems, itemsize = (next(args) for _ in range(4))
        call_args += [
            builder.load(primal.const),
            null_i8p,  # meminfo
            builder.load(primal.const),
            null_i8p,  # parent
            builder.load(primal.const),
            nitems,
            builder.load(primal.const),
            itemsize,
            builder.load(primal.dup),
            data,
            d_data,
        ]
        for _ in range(2 * spec.ndim):  # shapes, then strides
            call_args += [builder.load(primal.const), next(args)]
    return call_args


def _components(primal: _Primal, builder, pointer, flat: bool):
    """
    Yield a pointer to each component of one result value.

    Parameters
    ----------
    primal : _Primal
        Shared declarations of the driver module being built.
    builder : llvmlite.ir.IRBuilder
        Builder positioned in the emitting entry point's block.
    pointer : llvmlite.ir.Value
        Pointer to the result value.
    flat : bool
        Whether `pointer` addresses a caller-supplied buffer of components
        rather than an aggregate holding all of them.

    Yields
    ------
    llvmlite.ir.Value
        Pointer to one component, in order.

    See Also
    --------
    _sweep_prefix : Allocates the aggregates this addresses.
    """
    if primal.kernel.n_outputs == 1:
        yield pointer
        return
    for index in range(primal.kernel.n_outputs):
        leading = [] if flat else [ir.Constant(_I32, 0)]
        yield builder.gep(pointer, [*leading, ir.Constant(_I32, index)], inbounds=True)


def _sweep_prefix(primal: _Primal, builder, cotangent=None):
    """
    Allocate one sweep's result shadow and build its marker-call prefix.

    Numba's entry point writes through a return pointer and an
    exception-info pointer, so both are local to the entry point and only
    the return value's shadow is ever active.

    Parameters
    ----------
    primal : _Primal
        Shared declarations of the driver module being built.
    builder : llvmlite.ir.IRBuilder
        Builder positioned in the emitting entry point's block.
    cotangent : llvmlite.ir.Value, optional
        Pointer to the caller's output cotangent, copied into the shadow for
        a reverse sweep. Forward sweeps leave the shadow zeroed.

    Returns
    -------
    prefix : list of llvmlite.ir.Value
        The leading marker-call arguments both modes share.
    shadow : llvmlite.ir.Value
        Pointer to the result's shadow, holding the output tangent once a
        forward sweep has run.

    See Also
    --------
    _components : Addresses the shadow's individual components.
    """
    result = builder.alloca(primal.storage_type, name="result")
    builder.store(ir.Constant(primal.storage_type, None), result)
    shadow = builder.alloca(primal.storage_type, name="d_result")
    builder.store(ir.Constant(primal.storage_type, None), shadow)
    if cotangent is not None:
        for source, destination in zip(
            _components(primal, builder, cotangent, flat=True),
            _components(primal, builder, shadow, flat=False),
        ):
            builder.store(builder.load(source), destination)
    excinfo = builder.alloca(_EXCINFO_PTR_TYPE, name="excinfo")
    builder.store(ir.Constant(_EXCINFO_PTR_TYPE, None), excinfo)
    prefix = [
        builder.bitcast(primal.function, _I8P),
        builder.load(primal.dup),
        result,
        shadow,
        builder.load(primal.const),
        excinfo,
    ]
    return prefix, shadow


def _entry_function(primal: _Primal, symbol: str, return_type, params):
    """
    Declare one named entry point of a driver module.

    Parameters
    ----------
    primal : _Primal
        Shared declarations of the driver module being built.
    symbol : str
        Name of the entry point.
    return_type : llvmlite.ir.Type
        The entry point's return type.
    params : list of tuple of (str, llvmlite.ir.Type)
        Each parameter's name and LLVM type, in order.

    Returns
    -------
    llvmlite.ir.Function
        The declared entry point, its parameters named.
    """
    entry_fn = ir.Function(
        primal.module,
        ir.FunctionType(return_type, [param_type for _, param_type in params]),
        name=symbol,
    )
    for arg, (name, _) in zip(entry_fn.args, params):
        arg.name = name
    return entry_fn


def _synthesise_vjp(primal: _Primal) -> str:
    """
    Add the cotangent-seeded reverse-mode entry point to a driver module.

    The entry point is
    ``void vjp_<entry>(dx0_out, ..., cotangent, <arguments>)``: one output
    pointer per scalar argument, a pointer to the output cotangent, then the
    primal arguments. An array argument's gradient is accumulated in place
    into its caller-owned shadow buffer instead of an output pointer. A
    gradient is the same sweep seeded with one, so it needs no separate entry
    point.

    Parameters
    ----------
    primal : _Primal
        Shared declarations of the driver module being built.

    Returns
    -------
    str
        Name of the generated VJP symbol.

    See Also
    --------
    _synthesise_jvp : The forward-mode counterpart.
    """
    # Enzyme packs the gradients of the active by-value scalar arguments into
    # a literal struct, a bare scalar for one and nothing for none. Keep that
    # aggregate internal and expose one output pointer per scalar argument,
    # avoiding platform aggregate-return conventions and preserving each
    # argument's own derivative type.
    scalar_types = [
        _SCALAR_IR_TYPE[spec.llvm_type]
        for spec in primal.kernel.arg_specs
        if spec.kind == "scalar"
    ]
    if not scalar_types:
        gradient_type = ir.VoidType()
    elif len(scalar_types) == 1:
        gradient_type = scalar_types[0]
    else:
        gradient_type = ir.LiteralStructType(scalar_types)
    autodiff_fn = ir.Function(
        primal.module,
        ir.FunctionType(gradient_type, [_I8P], var_arg=True),
        name="__enzyme_autodiff",
    )

    n_outputs = len(scalar_types)
    scalar_indices = [
        index
        for index, spec in enumerate(primal.kernel.arg_specs)
        if spec.kind == "scalar"
    ]
    params = [
        (f"dx{index}_out", arg_type.as_pointer())
        for index, arg_type in zip(scalar_indices, scalar_types)
    ]
    params.append(("cotangent", primal.scalar_type.as_pointer()))
    params += _argument_params(primal.kernel, forward=False)
    symbol = f"vjp_{primal.kernel.entry_symbol}"
    entry_fn = _entry_function(primal, symbol, ir.VoidType(), params)

    builder = ir.IRBuilder(entry_fn.append_basic_block("entry"))
    prefix, _ = _sweep_prefix(primal, builder, cotangent=entry_fn.args[n_outputs])
    args = iter(entry_fn.args[n_outputs + 1 :])
    gradient = builder.call(
        autodiff_fn, [*prefix, *_marker_args(primal, builder, args, forward=False)]
    )
    for index, destination in enumerate(entry_fn.args[:n_outputs]):
        component = (
            gradient if n_outputs == 1 else builder.extract_value(gradient, index)
        )
        builder.store(component, destination)
    builder.ret_void()
    return symbol


def _synthesise_jvp(primal: _Primal) -> str:
    """
    Add the forward-mode entry point to a driver module.

    Every scalar argument is a ``(primal, tangent)`` duplicated pair, and an
    array argument's shadow slot is its caller-supplied tangent buffer. The
    product always lands in the result's shadow. A scalar primal returns it
    directly; a tuple-valued one copies it through a trailing ``out``
    pointer, since the shadow is an aggregate that would otherwise have to
    cross the external ABI boundary.

    Parameters
    ----------
    primal : _Primal
        Shared declarations of the driver module being built.

    Returns
    -------
    str
        Name of the generated JVP symbol.

    See Also
    --------
    _synthesise_vjp : The reverse-mode counterpart.
    """
    vector = primal.kernel.n_outputs > 1
    fwddiff_fn = ir.Function(
        primal.module,
        ir.FunctionType(primal.scalar_type, [_I8P], var_arg=True),
        name="__enzyme_fwddiff",
    )
    params = _argument_params(primal.kernel, forward=True)
    if vector:
        params.append(("out", primal.storage_type.as_pointer()))
    symbol = f"jvp_{primal.kernel.entry_symbol}"
    entry_fn = _entry_function(
        primal, symbol, ir.VoidType() if vector else primal.scalar_type, params
    )

    builder = ir.IRBuilder(entry_fn.append_basic_block("entry"))
    call_args, shadow = _sweep_prefix(primal, builder)
    call_args += _marker_args(primal, builder, iter(entry_fn.args), forward=True)
    builder.call(fwddiff_fn, call_args)

    if not vector:
        builder.ret(builder.load(shadow))
        return symbol
    builder.store(builder.load(shadow), entry_fn.args[-1])
    builder.ret_void()
    return symbol


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
        The driver module defining `jvp_<entry>` (forward mode) and
        `vjp_<entry>` (reverse mode) for `kernel`.

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
    >>> synthesise(lower(f)).vjp_symbol  # doctest: +SKIP
    'vjp__ZN...'
    """
    _, _, *flat_strs = kernel.arg_types
    scalar_type = _SCALAR_IR_TYPE[kernel.return_type]
    storage_type = (
        scalar_type
        if kernel.n_outputs == 1
        else ir.ArrayType(scalar_type, kernel.n_outputs)
    )

    module = ir.Module(name="numba_enzyme_driver")
    module.triple, module.data_layout = _target_lines(kernel.ir)

    kernel_fn = ir.Function(
        module,
        ir.FunctionType(
            ir.IntType(32),
            [
                storage_type.as_pointer(),
                _EXCINFO_PTR_TYPE.as_pointer(),
                *(_ir_type_from_str(s) for s in flat_strs),
            ],
        ),
        name=kernel.entry_symbol,
    )
    enzyme_dup = ir.GlobalVariable(module, ir.IntType(32), name="enzyme_dup")
    enzyme_dup.linkage = "external"
    enzyme_const = ir.GlobalVariable(module, ir.IntType(32), name="enzyme_const")
    enzyme_const.linkage = "external"

    primal = _Primal(
        kernel=kernel,
        module=module,
        function=kernel_fn,
        dup=enzyme_dup,
        const=enzyme_const,
        scalar_type=scalar_type,
        storage_type=storage_type,
    )
    jvp_symbol = _synthesise_jvp(primal)
    vjp_symbol = _synthesise_vjp(primal)
    return SynthesisedDriver(
        ir=str(module), jvp_symbol=jvp_symbol, vjp_symbol=vjp_symbol
    )
