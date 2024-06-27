import Flux
using Functors
using ChainRulesCore

@inline return_hidden_state(x::NamedTuple) = x
@inline return_hidden_state(x::T) where T = (hidden_state = x,)
@inline return_hidden_state(x, y) = return_hidden_state(return_hidden_state(x), return_hidden_state(y))
@inline return_hidden_state(x::NamedTuple, y) = return_hidden_state(x, return_hidden_state(y))
@inline return_hidden_state(x, y::NamedTuple) = return_hidden_state(return_hidden_state(x), y)
@inline function return_hidden_state(x::NamedTuple, y::NamedTuple)
    x_ext = Base.structdiff(x, y)
    hidden_state, y_ext = split_hidden_state(y)
    return merge(hidden_state, x_ext, y_ext)
end

split_hidden_state(nt::NamedTuple) =
    (haskey(nt, :hidden_state) ? (hidden_state = nt.hidden_state,) : (;)), Base.structdiff(nt, NamedTuple{(:hidden_state,)})

function ChainRulesCore.rrule(::typeof(split_hidden_state), nt::NamedTuple)
    hidden_state, ext = split_hidden_state(nt)
    function pullback(Ybar)
        Ȳ = unthunk(Ybar)
        ∂h = ChainRulesCore.backing(Ȳ[1])
        ∂ext = ChainRulesCore.backing(Ȳ[2])
        ∂nt = merge(∂h, ∂ext)
        return (NoTangent(), Tangent{Any, typeof(∂nt)}(∂nt))
    end
    return (hidden_state, ext), pullback
end

get_hidden_state(x::NamedTuple) = x.hidden_state
get_hidden_state(x) = x

#############################################

abstract type Architecture end
abstract type LayerStruct <: Architecture end
abstract type Layer{names} <: LayerStruct end

argument_names(::Layer{names}) where names = names
argument_names(_::A) where A >: Architecture = (:hidden_state,)
function argument_names(t::Tuple)
    if @generated
        N = fieldcount(t)
        expr = foldl([ :(argument_names(t[$n])) for n in 1:N ]) do init, ni
            :(Base.merge_names($init, $ni))
        end
        return expr
    else
        foldl((init,ti)->Base.merge_names(init, argument_names(ti)), t; init=())
    end
end
function argument_names(layer::LayerStruct)
    if @generated
        names = map(field->:(argument_names(layer.$field)), fieldnames(layer))
        expr = foldl((x,y)->:(Base.merge_names($x, $y)), names)
        return expr
    else
        return argument_names(ntuple(i->getfield(layer, i), Val(nfields(layer))))
    end
end

ChainRulesCore.@non_differentiable argument_names(m)

function return_namedtuple(f, nt, hidden_state)
    nt = Base.structdiff(nt, NamedTuple{argument_names(f)})
    return return_hidden_state(nt, hidden_state)
end

function (layer::LayerStruct)(arg, args...)
    arg isa NamedTuple && error("$(nameof(typeof(layer)))(::NamedTuple) is not overloaded.")
    names = argument_names(layer)
    input = tuple(arg, args...)
    length(names) != length(input) && error(
        "layer should take $(length(names)) arguments $names but only get $(length(input)).\nType: $(typeof(layer))"
    )
    return layer(NamedTuple{names}(input))
end

@inline apply_on_namedtuple(layer::L, nt::NamedTuple) where {L <: Architecture} = layer(nt)
@inline function apply_on_namedtuple(f, nt::NamedTuple)
    args = NamedTuple{argument_names(f)}(nt)
    y = f(args...)
    return return_namedtuple(f, nt, y)
end

with_extra(f, x, ext) = apply_on_namedtuple(f, return_hidden_state(ext, x))

#############################################

struct WithArg{names, L} <: Layer{names}
    layer::L
    function WithArg{names, L}(layer::L) where {names, L}
        @assert names isa Tuple{Vararg{Symbol}} "`WithArg{names}` where `names` must be tuple of symbols"
        return new{names, L}(layer)
    end
end
Functors.functor(::Type{<:WithArg{names}}, x) where names = ((layer = x.layer,), y -> WithArg{names}(y.layer))

WithArg(layer) = WithArg{(:hidden_state,)}(layer)
WithArg{names}(layer) where names = WithArg{names, typeof(layer)}(layer)

function (l::WithArg{names})(nt::NamedTuple) where names
    if @generated
        args = [:(nt.$n) for n in names]
        call = Expr(:call, :(l.layer), args...)
        return quote
            hidden_state = $call
            return return_namedtuple(l, nt, hidden_state)
        end
    else
        args = values(NamedTuple{argument_names(l)}(nt))
        hidden_state = l.layer(args...)
        return return_namedtuple(l, nt, hidden_state)
    end
end

function Base.show(io::IO, l::WithArg)
    print(io, "WithArg{", argument_names(l), "}(")
    show(io, l.layer)
    print(io, ')')
end
_show_name(l::WithArg) = join(("WithArg{", argument_names(l), "}"))

struct WithOptArg{names, opts, L} <: LayerStruct
    layer::L
    function WithOptArg{names, opts, L}(layer::L) where {names, opts, L}
        @assert names isa Tuple{Vararg{Symbol}} && opts isa Tuple{Vararg{Symbol}} "`WithOptArg{names, opts}` where `names` and `opts` must be tuple of symbols"
        return new{names, opts, L}(layer)
    end
end
Functors.functor(::Type{<:WithOptArg{names, opts}}, x) where {names, opts} =
    ((layer = x.layer,), y -> WithOptArg{names, opts}(y.layer))

WithOptArg{names, opts}(layer) where {names, opts} = WithOptArg{names, opts, typeof(layer)}(layer)

argument_names(::WithOptArg{names, opts}) where {names, opts} = (names..., opts...)

function (l::WithOptArg{names, opts})(nt::NamedTuple{input_names}) where {names, opts, input_names}
    if @generated
        arg_syms = Symbol[names...]
        for opt in opts
            sym_in(opt, input_names) == 0 && break
            push!(arg_syms, opt)
        end
        call = Expr(:call, :(l.layer), [ :(nt.$n) for n in arg_syms ]...)
        quote
            hidden_state = $call
            return return_namedtuple(l, nt, hidden_state)
        end
    else
        args = values(NamedTuple{argument_names(l)}(nt))
        for opt in opts
            !haskey(nt, opt) && break
            args = (args..., nt[opt])
        end
        hidden_state = l.layer(args...)
        return return_namedtuple(l, nt, hidden_state)
    end
end

function Base.show(io::IO, l::WithOptArg{names, opts}) where {names, opts}
    print(io, "WithOptArg{", names, ", ", opts, "}(")
    show(io, l.layer)
    print(io, ')')
end
_show_name(l::WithOptArg{names, opts}) where {names, opts} = join(("WithOptArg{", names, ", ", opts, "}"))

#############################################

struct RenameArgs{new_names, old_names, L} <: Architecture
    layer::L
    function RenameArgs{new_names, old_names, L}(layer::L) where {new_names, old_names, L}
        @assert new_names isa Tuple{Vararg{Symbol}} &&
            allunique(new_names) "`RenameArgs{new_names, old_names}` where `new_names` must be tuple of unique symbols"
        @assert old_names isa Tuple{Vararg{Symbol}} &&
            allunique(old_names) "`RenameArgs{new_names, old_names}` where `old_names` must be tuple of unique symbols"
        @assert length(new_names) == length(old_names) "`RenameArgs{new_names, old_names}` where `new_names` and `old_names` must have length"
        return new{new_names, old_names, L}(layer)
    end
end
Functors.functor(::Type{<:RenameArgs{new_names, old_names}}, x) where {new_names, old_names} =
    ((layer = x.layer,), y -> RenameArgs{new_names, old_names}(y.layer))

RenameArgs{new_names, old_names}(layer) where {new_names, old_names} =
    RenameArgs{new_names, old_names, typeof(layer)}(layer)

function argument_names(l::RenameArgs{new_names, old_names}) where {new_names, old_names}
    names = argument_names(l.layer)
    return replace_names(names, new_names, old_names)
end

function (l::RenameArgs{new_names, old_names})(nt::NamedTuple{names, types}) where {new_names, old_names, names, types}
    nt2 = NamedTuple{replace_names(names, old_names, new_names), types}(Tuple(nt))
    y = apply_on_namedtuple(l.layer, nt2)
    return return_hidden_state(nt, y)
end

function Base.show(io::IO, l::RenameArgs{new_names, old_names}) where {new_names, old_names}
    print(io, "RenameArgs{", old_names, " → ", new_names, "}(")
    show(io, l.layer)
    print(io, ')')
end
_show_name(l::RenameArgs{new_names, old_names}) where {new_names, old_names} = join(("RenameArgs{", old_names, " → ", new_names, "}"))

struct Branch{target, names, L} <: Architecture
    layer::L
    function Branch{target, names, L}(layer::L) where {target, names, L}
        @assert target isa Symbol ||
            target isa Tuple{Vararg{Symbol}} "`Branch{target}` where `target` must be a symbol or tuple of symbols"
        @assert names isa Tuple{Vararg{Symbol}} "`Branch{target, names}` where `names` must be tuple of symbols"
        return new{target, names, L}(layer)
    end
end
Functors.functor(::Type{<:Branch{target, names}}, x) where {target, names} =
    ((layer = x.layer,), y -> Branch{target, names}(y.layer))

argument_names(b::Branch{target, names}) where {target, names} = names isa Tuple{} ? argument_names(b.layer) : names

Branch{target}(layer) where target = Branch{target, (:hidden_state,)}(layer)
Branch{target, names}(layer) where {target, names} = Branch{target, names, typeof(layer)}(layer)

(b::Branch)(arg, args...) = b(NamedTuple{argument_names(b)}((arg, args...)))
function (b::Branch{target, names, L})(nt::NamedTuple) where {target, names, L}
    if @generated
        if names isa Tuple{}
            args = :nt
        else
            args = Expr(:call, :(NamedTuple{argument_names(b.layer)}), Expr(:tuple, [ :(nt.$n) for n in names ]...))
        end
        if target isa Symbol
            call = :(($target = y,))
        elseif target isa Tuple{}
            call = :y
        else
            call = :(NamedTuple{$target}(values(y)))
        end
        return quote
            y = apply_on_namedtuple(b.layer, $args)
            s = $call
            merge(nt, s)
        end
    else
        if names isa Tuple{}
            args = nt
        else
            args = NamedTuple{argument_names(b.layer)}(values(NamedTuple{names}(nt)))
        end
        y = apply_on_namedtuple(b.layer, args)
        if target isa Symbol
            s = NamedTuple{(target,)}((y,))
        elseif target isa Tuple{}
            s = y
        else
            s = NamedTuple{target}(values(y))
        end
        return merge(nt, s)
    end
end

function Base.show(io::IO, b::Branch{target}) where target
    print(io, "Branch{", target, " = ", argument_names(b), "}(")
    show(io, b.layer)
    print(io, ')')
end
_show_name(b::Branch{target}) where target = join(("Branch{", target, " = ", argument_names(b), "}"))

#############################################

struct Parallel{names, L}  <: Architecture
    layer::L
    function Parallel{names, L}(layer::L) where {names, L}
        @assert names isa Tuple{Vararg{Symbol}} "`Parallel{names}` where `names` must be tuple of symbols"
        return new{names, L}(layer)
    end
    function Parallel{names, L}(layer::L) where {names, L <: NamedTuple{names}}
        @assert names isa Tuple{Vararg{Symbol}} "`Parallel{names}` where `names` must be tuple of symbols"
        return new{names, L}(layer)
    end
end
Functors.functor(::Type{<:Parallel{names}}, x) where names = ((layer = x.layer,), y -> Parallel{names}(y.layer))

Parallel{names, L}(layer::L) where {names, L <: Tuple} = Parallel{names, NamedTuple{names, L}}(NamedTuple{names}(layer))
Parallel{names}(layer) where names = Parallel{names, typeof(layer)}(layer)

argument_names(p::Parallel{names}) where names = names

function (p::Parallel{names, L})(nt::NamedTuple) where {names, L}
    if @generated
        if L <: NamedTuple
            calls = [ :($n = apply_on_namedtuple(p.layer.$n, return_hidden_state(nt.$n))) for n in names ]
        else
            calls = [ :($n = apply_on_namedtuple(p.layer, return_hidden_state(nt.$n))) for n in names ]
        end
        expr = Expr(:tuple, calls...)
        return :(merge(nt, $expr))
    else
        if p.layer isa NamedTuple
            nts = apply_on_namedtuple.(values(p.layer), return_hidden_state.(values(NamedTuple{argument_names(p)}(nt))))
        else
            nts = map(Base.Fix1(apply_on_namedtuple, p.layer), return_hidden_state.(values(NamedTuple{argument_names(p)}(nt))))
        end
        return merge(nt, NamedTuple{argument_names(p)}(nts))
    end
end

function Base.show(io::IO, p::Parallel)
    print(io, "Parallel{", argument_names(p), "}(")
    show(io, p.layer)
    print(io, ')')
end
_show_name(p::Parallel) = join(("Parallel{", argument_names(p), "}"))

function applylayers(layers::NTuple{N, Any}, nt) where N
    if @generated
        symbols = [gensym() for _ in 1:N]
        pushfirst!(symbols, :nt)
        calls = [ :($(symbols[i+1]) = apply_on_namedtuple(layers[$i], $(symbols[i]))) for i in 1:N ]
        return Expr(:block, calls...)
    else
        return foldl((y, l) -> l(y), layers; init = nt)
    end
end

struct Chain{T<:Tuple} <: Architecture
    layers::T
end
@functor Chain

Chain(args...) = Chain{typeof(args)}(args)

function (c::Chain)(nt::NamedTuple)
    if @generated
        N = length(c.parameters[1].parameters)
        symbols = [gensym() for _ in 1:N]
        pushfirst!(symbols, :nt)
        calls = [ :($(symbols[i+1]) = apply_on_namedtuple(c.layers[$i], $(symbols[i]))) for i in 1:N ]
        return Expr(:block, calls...)
    else
        return applylayers(c.layers, nt)
    end
end
(c::Chain)(x) = c((hidden_state = x,))

Base.getindex(c::Chain, i) = c.layers[i]
Base.length(c::Chain) = length(c.layers)

function Base.show(io::IO, c::Chain)
    print(io, "Chain(")
    show(io, c[1])
    for ci in Base.tail(c.layers)
        print(io, ", ")
        show(io, ci)
    end
    print(io, ')')
end
Flux._show_children(c::Chain) = c.layers

#############################################

function Base.show(io::IO, m::MIME"text/plain", layer::Architecture)
    if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
        Flux._big_show(io, layer)
    elseif !get(io, :compact, false)
        Flux._layer_show(io, layer)
    else
        show(io, layer)
    end
end

_show_name(layer::Architecture) = nameof(typeof(layer))

function Flux._big_show(io::IO, layer::Architecture, indent::Int = 0, name = nothing)
    println(io, " "^indent, isnothing(name) ? "" : "$name = ", _show_name(layer), '(')
    for c in Flux._show_children(layer)
        Flux._big_show(io, c, indent + 2)
    end
    if iszero(indent)
        print(io, rpad(')', 2))
        Flux._big_finale(io, layer)
    else
        println(io, " "^indent, "),")
    end
end
