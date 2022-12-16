using StructWalk
using StructWalk: WalkStyle

abstract type LayerStyle <: WalkStyle end

StructWalk.walkstyle(::Type{LayerStyle}, x) = StructWalk._walkstyle(LayerStyle, x)
StructWalk.constructor(::Type{LayerStyle}, x) = StructWalk.constructor(WalkStyle, x)
StructWalk.children(::Type{LayerStyle}, x) = StructWalk.children(WalkStyle, x)
StructWalk.iscontainer(::Type{LayerStyle}, x) = StructWalk.iscontainer(WalkStyle, x)

StructWalk.children(::Type{LayerStyle}, ::AbstractArray) = ()
StructWalk.iscontainer(::Type{LayerStyle}, ::AbstractArray) = false
