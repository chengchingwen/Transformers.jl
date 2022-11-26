using Flux
using Flux: @functor
using MacroTools: @forward, postwalk

"""
    Stack(topo::NNTopo, layers...)

like Flux.Chain, but you can use a NNTopo to define the order/structure of the function called.
"""
struct Stack{FS, T<:Tuple}
    topo::NNTopo{FS}
    models::T
end

Stack(topo::NNTopo{FS}, xs...) where FS = Stack{FS, typeof(xs)}(topo, xs)

@functor Stack

@generated function (s::Stack{FS, TP})(xs...) where {FS, TP}
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

function Base.show(io::IO, s::Stack)
    print(io, "Stack(")
    join(io, s.models, ", ")
    print(io, ")")
end

"show the structure of the Stack function"
show_stackfunc(s::Stack) = print_topo(s.topo; models=s.models)
