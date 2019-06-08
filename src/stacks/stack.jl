using Flux
using MacroTools: @forward

"""
    Stack(topo::NNTopo, layers...)

like Flux.Chain, but you can use a NNTopo to define the order/structure of the function called.
"""
struct Stack{T<:Tuple, FS}
    models::T
    topo::NNTopo{FS}
    Stack(topo::NNTopo{FS}, xs...) where FS = new{typeof(xs), FS}(xs, topo)
end

Flux.children(s::Stack) = s.models
Flux.mapchildren(f, s::Stack) = Stack(s.topo, f.(s.models)...)

(s::Stack)(xs...) = s.topo(s.models, xs...)

@forward Stack.models Base.getindex, Base.length

"return a list of n model with give args"
stack(n, modeltype::DataType, args...; kwargs...) = [modeltype(args...; kwargs...) for i = 1:n]

function Base.show(io::IO, s::Stack)
    print(io, "Stack(")
    join(io, s.models, ", ")
    print(io, ")")
end

"show the structure of the Stack function"
show_stackfunc(s::Stack) = print_topo(s.topo; models=s.models)
