StructWalk.constructor(::Type{LayerStyle}, l::WithArg{names}) where names = WithArg{names}
StructWalk.constructor(::Type{LayerStyle}, l::WithOptArg{names, opts}) where {names, opts} = WithOptArg{names, opts}
StructWalk.constructor(::Type{LayerStyle}, l::RenameArgs{new_names, old_names}) where {new_names, old_names} =
    RenameArgs{new_names, old_names}
StructWalk.constructor(::Type{LayerStyle}, l::Branch{target, names}) where {target, names} = Branch{target, names}
StructWalk.constructor(::Type{LayerStyle}, l::Parallel{names}) where names = Parallel{names}

set_dropout(x, p) = x
set_dropout(dp::DropoutLayer, p) = DropoutLayer(dp.layer, p)
set_dropout(dp::Flux.Dropout, p) = Flux.Dropout(p, dp.dims, dp.active, dp.rng)

no_dropout(x) = set_dropout(x, nothing)
no_dropout(dp::Flux.Dropout) = Flux.Dropout(dp.p, dp.dims, false, dp.rng)

testmode(x) = postwalk(no_dropout, LayerStyle, x)
