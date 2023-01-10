StructWalk.constructor(::Type{LayerStyle}, l::WithArg{names}) where names = WithArg{names}
StructWalk.constructor(::Type{LayerStyle}, l::WithOptArg{names, opts}) where {names, opts} = WithOptArg{names, opts}
StructWalk.constructor(::Type{LayerStyle}, l::RenameArgs{new_names, old_names}) where {new_names, old_names} = RenameArgs{new_names, old_names}
StructWalk.constructor(::Type{LayerStyle}, l::Branch{target, names}) where {target, names} = Branch{target, names}
StructWalk.constructor(::Type{LayerStyle}, l::Parallel{names}) where names = Parallel{names}

set_dropout(x, p) = x
set_dropout(dp::DropoutLayer, p) = DropoutLayer(dp.layer, p)

no_dropout(x) = set_dropout(x, nothing)

testmode(x) = postwalk(no_dropout, LayerStyle, x)
