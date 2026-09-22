"""
Synthesise the Enzyme driver module for a lowered kernel.

Builds a second LLVM IR module, via :mod:`llvmlite.ir`'s typed builder
API rather than templated C source, declaring the Numba kernel with its
exact discovered type (no bitcast needed) and defining the reverse-mode
(``__enzyme_autodiff``) and forward-mode (``__enzyme_fwddiff``) entry
points with the correct per-argument activity markers.

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

from numba_enzyme.lowering import LoweredKernel

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

# TODO: expand for other scalar types
_SCALAR_IR_TYPE = {
    "double": ir.DoubleType(),
    "float": ir.FloatType(),
    "i32": ir.IntType(32),
    "i64": ir.IntType(64),
}


@dataclass(frozen=True)
class SynthesisedDriver:
    """
    The Enzyme driver module built for one `LoweredKernel`.

    Attributes
    ----------
    ir : str
        The driver module's full LLVM IR text, ready to be linked
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
    arg_types : tuple of llvmlite.ir.Type
        LLVM type of each scalar primal argument.
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
    arg_types: tuple
    scalar_type: ir.Type
    storage_type: ir.Type


def _target_lines(kernel_ir: str) -> tuple[str, str]:
    """
    Extract the target triple and datalayout from Numba's own IR.

    Parameters
    ----------
    kernel_ir : str
        The kernel's LLVM IR text, as produced by
        `numba_enzyme.lowering.lower`.

    Returns
    -------
    triple : str
        The ``target triple`` string.
    datalayout : str
        The ``target datalayout`` string.

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
    # TODO: instantiate them as empty string
    triple = datalayout = None
    for line in kernel_ir.splitlines():
        if line.startswith("target triple"):
            triple = line.split('"')[1]
        elif line.startswith("target datalayout"):
            datalayout = line.split('"')[1]
    return triple, datalayout


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


def _synthesise_vjp(primal: _Primal, autodiff_fn) -> str:
    """
    Add the cotangent-seeded reverse-mode entry point to a driver module.

    The entry point is ``void vjp_<entry>(dx0_out, ..., cotangent, x0, ...)``
    for every arity and result shape. A gradient is the same sweep seeded
    with one, so it needs no separate entry point.

    Parameters
    ----------
    primal : _Primal
        Shared declarations of the driver module being built.
    autodiff_fn : llvmlite.ir.Function
        Declaration of Enzyme's reverse-mode marker.

    Returns
    -------
    str
        Name of the generated VJP symbol.

    See Also
    --------
    _synthesise_jvp : The forward-mode counterpart.
    """
    n_args = len(primal.arg_types)
    symbol = f"vjp_{primal.kernel.entry_symbol}"
    entry_fn = ir.Function(
        primal.module,
        ir.FunctionType(
            ir.VoidType(),
            [
                *(arg_type.as_pointer() for arg_type in primal.arg_types),
                primal.scalar_type.as_pointer(),
                *primal.arg_types,
            ],
        ),
        name=symbol,
    )
    for index in range(n_args):
        entry_fn.args[index].name = f"dx{index}_out"
    entry_fn.args[n_args].name = "cotangent"
    xs = entry_fn.args[n_args + 1 :]
    for index, arg in enumerate(xs):
        arg.name = f"x{index}"

    builder = ir.IRBuilder(entry_fn.append_basic_block("entry"))
    prefix, _ = _sweep_prefix(primal, builder, cotangent=entry_fn.args[n_args])
    gradient = builder.call(autodiff_fn, [*prefix, *xs])
    for index, destination in enumerate(entry_fn.args[:n_args]):
        component = gradient if n_args == 1 else builder.extract_value(gradient, index)
        builder.store(component, destination)
    builder.ret_void()
    return symbol


def _synthesise_jvp(primal: _Primal) -> str:
    """
    Add the forward-mode entry point to a driver module.

    Every active argument is a ``(primal, tangent)`` duplicated pair and the
    product always lands in the result's shadow. A scalar primal returns it
    directly; a tuple-valued one writes its components through a leading
    ``out`` pointer, since the shadow is an aggregate that would otherwise
    have to cross the external ABI boundary.

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
    n_args = len(primal.arg_types)
    vector = primal.kernel.n_outputs > 1
    fwddiff_fn = ir.Function(
        primal.module,
        ir.FunctionType(primal.scalar_type, [_I8P], var_arg=True),
        name="__enzyme_fwddiff",
    )
    symbol = f"jvp_{primal.kernel.entry_symbol}"
    pairs = [item for arg_type in primal.arg_types for item in (arg_type, arg_type)]
    entry_fn = ir.Function(
        primal.module,
        ir.FunctionType(
            ir.VoidType() if vector else primal.scalar_type,
            ([primal.scalar_type.as_pointer()] if vector else []) + pairs,
        ),
        name=symbol,
    )
    offset = 1 if vector else 0
    if vector:
        entry_fn.args[0].name = "out"
    for index in range(n_args):
        entry_fn.args[offset + 2 * index].name = f"x{index}"
        entry_fn.args[offset + 2 * index + 1].name = f"dx{index}"

    builder = ir.IRBuilder(entry_fn.append_basic_block("entry"))
    call_args, shadow = _sweep_prefix(primal, builder)
    for index in range(n_args):
        call_args.extend(
            [
                builder.load(primal.dup),
                entry_fn.args[offset + 2 * index],
                entry_fn.args[offset + 2 * index + 1],
            ]
        )
    builder.call(fwddiff_fn, call_args)

    if not vector:
        builder.ret(builder.load(shadow))
        return symbol
    for source, destination in zip(
        _components(primal, builder, shadow, flat=False),
        _components(primal, builder, entry_fn.args[0], flat=True),
    ):
        builder.store(builder.load(source), destination)
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
        The driver module defining ``jvp_<entry>`` (forward mode) and
        ``vjp_<entry>`` (reverse mode) for `kernel`.

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
    _, _, *scalar_strs = kernel.arg_types
    scalar_type = _SCALAR_IR_TYPE[kernel.return_type]
    arg_types = tuple(_SCALAR_IR_TYPE[name] for name in scalar_strs)
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
                *arg_types,
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
        arg_types=arg_types,
        scalar_type=scalar_type,
        storage_type=storage_type,
    )

    # Enzyme packs gradients of active by-value arguments into a literal
    # struct. Keep that aggregate internal and expose one output pointer per
    # argument, avoiding platform aggregate-return conventions and preserving
    # each argument's own derivative type.
    grad_ret_type = (
        arg_types[0] if len(arg_types) == 1 else ir.LiteralStructType(list(arg_types))
    )
    autodiff_fn = ir.Function(
        module,
        ir.FunctionType(grad_ret_type, [_I8P], var_arg=True),
        name="__enzyme_autodiff",
    )

    jvp_symbol = _synthesise_jvp(primal)
    vjp_symbol = _synthesise_vjp(primal, autodiff_fn)
    return SynthesisedDriver(
        ir=str(module), jvp_symbol=jvp_symbol, vjp_symbol=vjp_symbol
    )
