using Flux
using MacroTools: @forward, postwalk

"""
    Stack(topo::NNTopo, layers...)

like Flux.Chain, but you can use a NNTopo to define the order/structure of the function called.
"""
struct Stack{T<:Tuple, FS}
    models::T
    topo::NNTopo{FS}
    Stack(topo::NNTopo{FS}, xs...) where FS = new{typeof(xs), FS}(xs, topo)
end

Flux.functor(s::Stack) = s.models, m -> Stack(s.topo, m...)

@generated function (s::Stack{TP, FS})(xs...) where {TP, FS}
    _code = nntopo_impl(FS)
    n = fieldcount(TP)
    ms = [Symbol(:__model, i, :__) for i = 1:n]
    head = Expr(:(=), Expr(:tuple, ms...), :(s.models))
    pushfirst!(_code.args, head)
    code = postwalk(_code) do x
        if x isa Expr && x.head === :ref && x.args[1] === :model
            i = x.args[2]
            y = :($(ms[i]))
            return y
        else
            x
        end
    end
    return code
end

@forward Stack.models Base.getindex, Base.length

"return a list of n model with give args"
stack(n, modeltype::Type{T}, args...; kwargs...) where T = [modeltype(args...; kwargs...) for i = 1:n]

function Base.show(io::IO, s::Stack)
    print(io, "Stack(")
    join(io, s.models, ", ")
    print(io, ")")
end

"show the structure of the Stack function"
show_stackfunc(s::Stack) = print_topo(s.topo; models=s.models)
