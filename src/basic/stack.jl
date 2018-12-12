using Flux

struct Stack{T<:Tuple}
    models::T
    t::NetTopo
    Stack(t, xs...) = new{typeof(xs)}(xs, t)
end

Stack(n::Int, x) = Stack([deepcopy(x) for i = 1:n]...)


Flux.children(s::Stack) = s.models
Flux.mapchildren(f, s::Stack) = Stack(f.(s.models)...)


(s::Stack)(xs...) = s.t(s.models, xs...)
