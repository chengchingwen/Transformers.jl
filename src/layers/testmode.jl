StructWalk.constructor(::Type{LayerStyle}, l::WithArg{names}) where names = WithArg{names}
StructWalk.constructor(::Type{LayerStyle}, l::WithOptArg{names, opts}) where {names, opts} = WithOptArg{names, opts}
StructWalk.constructor(::Type{LayerStyle}, l::RenameArgs{new_names, old_names}) where {new_names, old_names} =
    RenameArgs{new_names, old_names}
StructWalk.constructor(::Type{LayerStyle}, l::Branch{target, names}) where {target, names} = Branch{target, names}
StructWalk.constructor(::Type{LayerStyle}, l::Parallel{names}) where names = Parallel{names}

no_dropout_overload(x) = methods(set_dropout, Tuple{typeof(x), Any}).ms[].sig.types[2] >: Any

set_dropout(x, p) = postwalk(LayerStyle, x) do xi
    no_dropout_overload(xi) ? xi : set_dropout(xi, p)
end
set_dropout(dp::DropoutLayer, p) = DropoutLayer(dp.layer, p)
set_dropout(dp::Flux.Dropout, p) = Flux.Dropout(p, dp.dims, dp.active, dp.rng)
set_dropout(dp::Flux.Dropout, ::Nothing) = Flux.Dropout(dp.p, dp.dims, false, dp.rng)

no_dropout(x) = set_dropout(x, nothing)

"""
    testmode(model)

Creating a new model sharing all parameters with `model` but used for testing. Currently this is just
 [`no_dropout`](@ref).
"""
testmode(x) = no_dropout(x)

"""
    set_dropout(model, p)

Creating a new model sharing all parameters with `model` but set all dropout probability to `p`.
"""
set_dropout

"""
    no_dropout(model)

Creating a new model sharing all parameters with `model` but disable all dropout.
"""
no_dropout
