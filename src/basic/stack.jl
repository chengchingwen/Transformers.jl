using Flux

struct Stack{T<:Tuple}
    models::T
    topo::NNTopo
    Stack(topo, xs...) = new{typeof(xs)}(xs, topo)
end


# Stack(n::Int, x; topo::String = "") = Stack(n, x, topo == "" ? NNTopo("x => $n") : NNTopo(topo))
# function Stack(n::Int, x, topo::NNTopo)
#     models = stack(n, x)
#     Stack(topo, models...)
# end

Flux.children(s::Stack) = s.models
Flux.mapchildren(f, s::Stack) = Stack(s.topo, f.(s.models)...)

(s::Stack)(xs...) = s.topo(s.models, xs...)

"return a list of n model with give args"
stack(n, modeltype::DataType, args...) = [modeltype(args...) for i = 1:n]

function Base.show(io::IO, s::Stack)
    print(io, "Stack(")
    join(io, s.models, ", ")
    print(io, ")")
end

show_stackfunc(s::Stack) = print_topo(s.topo; models=s.models)
