import Flux
using NNlib
using Functors
using ChainRulesCore
using Static

using NeuralAttentionlib
using NeuralAttentionlib: $, AbstractAttenOp, MultiheadQKVAttenOpWithScore,
    MultiheadQKVAttenOp, CausalMultiheadQKVAttenOp, CausalMultiheadQKVAttenOpWithScore

@static if VERSION < v"1.8"
    macro etotal(ex)
        return :(Base.@pure $ex)
    end
else
    macro etotal(ex)
        return :(Base.@assume_effects :total $ex)
    end
end

@etotal function sym_in(x::Symbol, xs::Tuple{Vararg{Symbol}})
    @nospecialize xs
    for i = 1:length(xs)
        x == xs[i] && return i
    end
    return 0
end

@etotal function prefix_name(prefix::Symbol, names::Tuple{Vararg{Symbol}})
    @nospecialize names
    return map(Base.Fix1(Symbol, Symbol(prefix, :_)), names)
end

@etotal function replace_name(names::Tuple{Vararg{Symbol}}, a::Symbol, b::Symbol)
    @nospecialize names
    return map(name -> name == a ? b : name, names)
end

@etotal function replace_names(names::Tuple{Vararg{Symbol}}, as::NTuple{N, Symbol}, bs::NTuple{N, Symbol}) where N
    @nospecialize names as bs
    for i in Base.OneTo(N)
        names = replace_name(names, as[i], bs[i])
    end
    return names
end

@etotal function remove_name(names::Tuple{Vararg{Symbol}}, name::Symbol)
    @nospecialize names
    i = sym_in(name, names)
    return i == 0 ? names : (names[1:i-1]..., names[i+1:end]...)
end

function rename(nt::NamedTuple{names, types}, _a::Val{a}, _b::Val{b}) where {names, types, a, b}
    if iszero(sym_in(b, names))
        new_names = replace_name(names, a, b)
        return NamedTuple{new_names, types}(values(nt))
    else
        nt = Base.structdiff(nt, NamedTuple{(b,)})
        return rename(nt, _a, _b)
    end
end

function with_prefix(::Val{prefix}, nt::NamedTuple{names, types}) where {prefix, names, types}
    new_names = prefix_name(prefix, names)
    return NamedTuple{new_names, types}(values(nt))
end
with_prefix(prefix::Val) = Base.Fix1(with_prefix, prefix)

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
argument_names(_) = (:hidden_state,)
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

function return_namedtuple(f, nt, hidden_state)
    nt = Base.structdiff(nt, NamedTuple{argument_names(f)})
    return return_hidden_state(nt, hidden_state)
end

function (layer::LayerStruct)(arg, args...)
    arg isa NamedTuple && error("$(typeof(layer))(::NamedTuple) is not overloaded.")
    names = argument_names(layer)
    input = tuple(arg, args...)
    length(names) != length(input) && error(
        "$(typeof(layer)) should take $(length(names)) arguments but only get $(length(input))"
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
    return replace_names(names, old_names, new_names)
end

function (l::RenameArgs{new_names, old_names})(nt::NamedTuple{names, types}) where {new_names, old_names, names, types}
    nt2 = NamedTuple{replace_names(names, old_names, new_names), types}(Tuple(nt))
    y = apply_on_namedtuple(l.layer, nt2)
    return return_hidden_state(nt, y)
end

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
            call = y
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
    print(io, "Branch{", target, ", ", argument_names(b), "}(")
    show(io, b.layer)
    print(io, ')')
end

#############################################

struct Parallel{names, L}  <: Architecture
    layer::L
    function Parallel{names, L}(layer::L) where {names, L}
        @assert names isa Tuple{Vararg{Symbol}} "`Parallel{names}` where `names` must be tuple of symbols"
        return new{names, L}(layer)
    end
end
Functors.functor(::Type{<:Parallel{names}}, x) where names = ((layer = x.layer,), y -> Parallel{names}(y.layer))

Parallel{names}(layer) where names = Parallel{names, typeof(layer)}(layer)

function (p::Parallel{names})(nt::NamedTuple) where names
    if @generated
        calls = [ :($n = apply_on_namedtuple(p.layer, return_hidden_state(nt.$n))) for n in names ]
        expr = Expr(:tuple, calls...)
        return :(merge(nt, $expr))
    else
        nts = map(Base.Fix1(apply_on_namedtuple, p.layer), return_hidden_state.(values(NamedTuple{argument_names(p)}(nt))))
        return merge(nt, NamedTuple{argument_names(p)}(nts))
    end
end

function Base.show(io::IO, p::Parallel)
    print(io, "Parallel{", argument_names(p), "}(")
    show(io, p.layer)
    print(io, ')')
end

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

#############################################

struct Fork{T<:Tuple}
    layers::T
end
Fork(layers...) = Fork(layers)

@functor Fork

function (f::Fork)(x)
    return ntuple(i -> f.layers[i](x), Val(length(f.layers)))
end

struct NSplit{N, L}
    n::N
    layer::L
end
NSplit(n::Integer, layer) = NSplit(static(n), layer)

@functor NSplit (layer,)

function nsplit(x, hdim, i)
    b = hdim * i
    a = b - hdim + 1
    cs = ntuple(i->Colon(), static(ndims(x)) - static(1))
    return @view x[a:b, cs...]
end

function (ns::NSplit)(x)
    y = ns.layer(x)
    ndim = ndims(y)
    hdim, r = divrem(size(y, 1), ns.n)
    @assert iszero(r) "NSplit try to split $(size(y,1)) in to $(Int(ns.n)) tensors"
    return ntuple(nsplit $ y $ hdim, ns.n)
end

bias_and_act(act, b, x) = act.(x .+ b)
bias_and_act(::Nothing, b, x) = x .+ b
bias_and_act(act, ::Nothing, x) = act.(x)
bias_and_act(::Nothing, ::Nothing, x) = x

dense(act, W, b, x) = bias_and_act(act, b, NeuralAttentionlib.scaled_matmul(W, x))

struct Dense{F, T, B}
    σ::F
    W::T
    b::B
end
@functor Dense (W, b)

Dense(w::AbstractArray) = Dense(nothing, w, nothing)
Dense(act, w::AbstractArray) = Dense(act, w, nothing)
Dense(w::AbstractArray, b::AbstractArray) = Dense(nothing, w, b)

Dense(din::Int, dout::Int; bias = true) = Dense(nothing, din, dout; bias)
function Dense(act, din::Int, dout::Int; bias = true)
    b = bias ? Flux.glorot_uniform(dout) : nothing
    w = Flux.glorot_uniform(dout, din)
    return Dense(act, w, b)
end

(d::Dense)(x) = dense(d.σ, d.W, d.b, x)

function Base.show(io::IO, d::Dense)
    print(io, "Dense(")
    if !isnothing(d.σ)
        print(io, "σ = ")
        show(io, d.σ)
        print(io, ", ")
    end
    print(io, "W = ")
    print(io, reverse(size(d.W)))
    print(io, ", b = ")
    print(io, !isnothing(d.b))
    print(io, ')')
end

struct LayerNorm{A, B, F}
    α::A
    β::B
    ϵ::F
end
@functor LayerNorm (α, β)

(ln::LayerNorm)(x) = NeuralAttentionlib.layer_norm(ln.ϵ, ln.α, ln.β, x)

LayerNorm(hidden_size::Int; ϵ = 1e-7) = LayerNorm(ones(Float32, hidden_size), zeros(Float32, hidden_size), Float32(ϵ))

Base.show(io::IO, ln::LayerNorm) = print(io, "LayerNorm(", length(ln.α), ", ϵ = ", ln.ϵ, ')')

struct RMSLayerNorm{A, F}
    α::A
    ϵ::F
end
@functor RMSLayerNorm (α,)

(ln::RMSLayerNorm)(x) = NeuralAttentionlib.rms_layer_norm(ln.ϵ, ln.α, x)

RMSLayerNorm(hidden_size::Int; ϵ = 1e-7) = RMSLayerNorm(ones(Float32, hidden_size), Float32(ϵ))

Base.show(io::IO, ln::RMSLayerNorm) = print(io, "RMSLayerNorm(", length(ln.α), ", ϵ = ", ln.ϵ, ')')

struct DropoutLayer{L, P} <: LayerStruct
    layer::L
    p::P
end
@functor DropoutLayer (layer,)

argument_names(dp::DropoutLayer) = argument_names(dp.layer)

function (dp::DropoutLayer)(nt::NamedTuple)
    y = apply_on_namedtuple(dp.layer, nt)
    if isnothing(dp.p)
        return y
    else
        hidden_state = NeuralAttentionlib.dropout(y.hidden_state, dp.p)
        return return_hidden_state(y, hidden_state)
    end
end

#############################################

abstract type AbstractTransformerBlock <: LayerStruct end

struct TransformerBlock{A, F} <: AbstractTransformerBlock
    attention::A
    feedforward::F
end
@functor TransformerBlock

(b::TransformerBlock)(nt::NamedTuple) = apply_on_namedtuple(b.feedforward, apply_on_namedtuple(b.attention, nt))

struct TransformerDecoderBlock{A, C, F} <: AbstractTransformerBlock
    attention::A
    crossattention::C
    feedforward::F
end
@functor TransformerDecoderBlock

argument_names(b::TransformerDecoderBlock) = Base.merge_names(
    Base.merge_names(argument_names(b.crossattention), argument_names(b.attention)),
    argument_names(b.feedforward)
)

(b::TransformerDecoderBlock)(nt::NamedTuple) =
    apply_on_namedtuple(b.feedforward, apply_on_namedtuple(b.crossattention, apply_on_namedtuple(b.attention, nt)))

struct PreNormResidual{L, N} <: LayerStruct
    layer::L
    norm::N
end
@functor PreNormResidual

function (prenr::PreNormResidual)(nt::NamedTuple)
    norm = apply_on_namedtuple(prenr.norm, nt)
    y = apply_on_namedtuple(prenr.layer, norm)
    hidden_state = y.hidden_state + nt.hidden_state
    return return_hidden_state(y, hidden_state)
end

struct PostNormResidual{L, N} <: LayerStruct
    layer::L
    norm::N
end
@functor PostNormResidual

function (postnr::PostNormResidual)(nt::NamedTuple)
    y = apply_on_namedtuple(postnr.layer, nt)
    hidden_state = y.hidden_state + nt.hidden_state
    r = return_hidden_state(y, hidden_state)
    return apply_on_namedtuple(postnr.norm, r)
end

const  PreNormTransformerBlock{A, LN1, F, LN2} = TransformerBlock{ PreNormResidual{A, LN1},  PreNormResidual{F, LN2}}
const PostNormTransformerBlock{A, LN1, F, LN2} = TransformerBlock{PostNormResidual{A, LN1}, PostNormResidual{F, LN2}}
const  PreNormTransformerDecoderBlock{A, LN1, C, LN2, F, LN3} =
    TransformerDecoderBlock{ PreNormResidual{A, LN1},  PreNormResidual{C, LN2},  PreNormResidual{F, LN3}}
const PostNormTransformerDecoderBlock{A, LN1, C, LN2, F, LN3} =
    TransformerDecoderBlock{PostNormResidual{A, LN1}, PostNormResidual{C, LN2}, PostNormResidual{F, LN3}}

#############################################

argument_names(::MultiheadQKVAttenOpWithScore) = (:hidden_state, :attention_mask)
argument_names(::MultiheadQKVAttenOp) = (:hidden_state, :attention_mask)
argument_names(::CausalMultiheadQKVAttenOp) = (:hidden_state, :attention_mask)
argument_names(::CausalMultiheadQKVAttenOpWithScore) = (:hidden_state, :attention_mask)

function apply_attention_op(op, nt::NamedTuple)
    qkv = nt.hidden_state
    qkv isa NTuple{3, Any} ||
        error("Expect hidden_state to be a tuple of 3 arrays, but get $(typeof(qkv)).")
    mask = get(nt, :attention_mask, nothing)
    a = op(qkv..., mask)
    return return_hidden_state(nt, a)
end

function apply_on_namedtuple(
    op::Union{MultiheadQKVAttenOpWithScore, MultiheadQKVAttenOp,
              CausalMultiheadQKVAttenOp, CausalMultiheadQKVAttenOpWithScore},
    nt::NamedTuple
)
    return apply_attention_op(op, nt)
end

struct SelfAttention{A, QKV, O} <: LayerStruct
    attention_op::A
    qkv_proj::QKV #::NSplit{StaticInt{3}, QKV}
    o_proj::O
end
@functor SelfAttention

function (sa::SelfAttention)(nt::NamedTuple)
    qkv = apply_on_namedtuple(sa.qkv_proj, nt)
    a = apply_on_namedtuple(sa.attention_op, qkv)
    y = apply_on_namedtuple(sa.o_proj, a)
    return y
end

struct CrossAttention{A, Q, KV, O} <: LayerStruct
    attention_op::A
    q_proj::Q
    kv_proj::KV #::NSplit{StaticInt{2}, KV}
    o_proj::O
end
@functor CrossAttention

function argument_names(ca::CrossAttention)
    required_names = (:hidden_state, :memory)
    field_names = invoke(argument_names, Tuple{LayerStruct}, ca)
    cross_field_names = remove_name(prefix_name(:cross, field_names), :cross_hidden_state)
    return Base.merge_names(required_names, cross_field_names)
end

function (ca::CrossAttention)(nt::NamedTuple)
    hidden_state, memory = nt.hidden_state, nt.memory
    cross_attention_mask = get(nt, :cross_attention_mask, nothing)
    nt_ext = Base.structdiff(nt, NamedTuple{(:hidden_state, :memory, :attention_mask, :cross_attention_mask)})
    q = with_extra(ca.q_proj, hidden_state, nt_ext)
    kv = with_extra(ca.kv_proj, memory, nt_ext)
    kv.hidden_state isa NTuple{2, Any} ||
        error("Expect kv_proj(memory).hidden_state return a tuple of 2 arrays, but get $(typeof(kv.hidden_state)).")
    qkv = merge(kv, q, (
        hidden_state = (q.hidden_state, kv.hidden_state...),
        attention_mask = cross_attention_mask,
    ))
    _a = apply_on_namedtuple(ca.attention_op, Base.structdiff(qkv, NamedTuple{(:attention_score,)}))
    a = rename(Base.structdiff(_a, NamedTuple{(:attention_mask, :cross_attention_mask)}),
               Val(:attention_score), Val(:cross_attention_score))
    y = apply_on_namedtuple(ca.o_proj, a)
    return merge(nt, y)
end

#############################################

struct Transformer{T <: Tuple{Vararg{<:AbstractTransformerBlock}}, F} <: LayerStruct
    blocks::T
    f::F
end
Transformer(blocks::Tuple{Vararg{AbstractTransformerBlock}}) = Transformer(blocks, nothing)
Transformer(blocks::AbstractTransformerBlock...) = Transformer(blocks)

@functor Transformer

(t::Transformer)(nt::NamedTuple) = applyblocks(t.blocks, t.f, nt)

function _block_call(symbols, i, has_f)
    call = :(blocks[$i]($(symbols[i])))
    if has_f
        call = :(f($(symbols[i]), $call))
    end
    line = :($(symbols[i+1]) = $call)
    return line
end

function applyblocks(blocks::Tuple{Vararg{AbstractTransformerBlock, N}}, f, x) where N
    if @generated
        symbols = vcat(:x, [gensym() for _ in 1:N])
        has_f = !(f <: Nothing)
        calls = [ _block_call(symbols, i, has_f) for i in 1:N ]
        return Expr(:block, calls...)
    else
        if isnothing(f)
            return foldl((y, blk)-> blk(y), blocks; init=x)
        else
            return foldl((y, blk)-> f(y, blk(y)), blocks; init=x)
        end
    end
end

Base.getindex(t::Transformer, i::Integer) = t.blocks[i]
Base.getindex(t::Transformer, r::AbstractVector) = Transformer(t.blocks[r])
Base.length(t::Transformer) = length(t.blocks)

function Transformer(T::Type{<:AbstractTransformerBlock}, n::Int, args...; collect_outputs = false, kwargs...)
    collect_f = collect_outputs isa Bool ?
        (collect_outputs ? (@__MODULE__).collect_outputs : nothing) :
        collect_outputs
    return Transformer(ntuple(i -> T(args...; kwargs...), n), collect_f)
end

#############################################

function collect_outputs(prev, output)
    hidden_state = output.hidden_state
    if haskey(prev, :outputs)
        prev_outputs = prev.outputs
        new_output = NamedTuple{keys(first(prev_outputs))}(output) # assume each block give the same outputs
        outputs = (prev_outputs..., new_output)
    else
        new_output = Base.structdiff(output, prev)
        outputs = (merge((hidden_state = hidden_state,), new_output),)
    end
    return merge(output, (outputs = outputs,))
end

#############################################

function SelfAttention(head::Int, hidden_size::Int; dropout = nothing, return_score = false, causal = false)
    @assert rem(hidden_state, head) == 0 "`hidden_size` should be dividible by `head` if `head_hidden_size` is not set"
    head_hidden_size = div(hidden_size, head)
    return SelfAttention(head, hidden_size, head_hidden_size; dropout, return_score, causal)
end
function SelfAttention(
    head::Int, hidden_size::Int, head_hidden_size::Int;
    dropout = nothing, return_score = false, causal = false,
)
    atten_op_constr = causal ?
        (return_score ? CausalMultiheadQKVAttenOpWithScore : CausalMultiheadQKVAttenOp) :
        (return_score ? MultiheadQKVAttenOpWithScore : MultiheadQKVAttenOp)
    atten_op = atten_op_constr(head, dropout)
    qkv_proj = Flux.Dense(hidden_size, 3head * head_hidden_size)
    o_proj = Flux.Dense(head * head_hidden_size, hidden_size)
    sa = SelfAttention(atten_op, NSplit(static(3), qkv_proj), o_proj)
    return sa
end

function CrossAttention(head::Int, hidden_size::Int; dropout = nothing, return_score = false)
    @assert rem(hidden_state, head) == 0 "`hidden_size` should be dividible by `head` if `head_hidden_size` is not set"
    head_hidden_size = div(hidden_size, head)
    return CrossAttention(head, hidden_size, head_hidden_size; dropout, return_score)
end
function CrossAttention(head::Int, hidden_size::Int, head_hidden_size::Int; dropout = nothing, return_score = false)
    cross_atten_op = return_score ?
        MultiheadQKVAttenOpWithScore(head, dropout) :
        MultiheadQKVAttenOp(head, dropout)
    c_q_proj = Flux.Dense(hidden_size, head * head_hidden_size)
    c_kv_proj = Flux.Dense(hidden_size, 2head * head_hidden_size)
    c_o_proj = Flux.Dense(head * head_hidden_size, hidden_size)
    ca = CrossAttention(cross_atten_op, c_q_proj, NSplit(static(2), c_kv_proj), c_o_proj)
    return ca
end

#############################################

TransformerBlock(
    head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    attention_dropout = nothing, dropout = nothing, return_score = false,
) = TransformerBlock(
    gelu, head, hidden_size, head_hidden_size, intermediate_size; attention_dropout, dropout, return_score)

TransformerBlock(
    act, head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    attention_dropout = nothing, dropout = nothing, return_score = false,
) = PostNormTransformerBlock(
    act, head, hidden_size, head_hidden_size, intermediate_size; attention_dropout, dropout, return_score)

PostNormTransformerBlock(
    head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    attention_dropout = nothing, dropout = nothing, return_score = false,
) = PostNormTransformerBlock(
    gelu, head, hidden_size, head_hidden_size, intermediate_size; attention_dropout, dropout, return_score)

function PostNormTransformerBlock(
    act, head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    attention_dropout = nothing, dropout = nothing, return_score = false,
)
    sa = SelfAttention(head, hidden_size, head_hidden_size; dropout = attention_dropout, return_score)
    ff1 = Flux.Dense(hidden_size, intermediate_size, act)
    ff2 = Flux.Dense(intermediate_size, hidden_size)
    return TransformerBlock(
        PostNormResidual(
            DropoutLayer(sa, dropout),
            LayerNorm(hidden_size)),
        PostNormResidual(
            DropoutLayer(Flux.Chain(ff1, ff2), dropout),
            LayerNorm(hidden_size)))
end

PreNormTransformerBlock(
    head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    attention_dropout = nothing, dropout = nothing, return_score = false,
) = PreNormTransformerBlock(
    gelu, head, hidden_size, head_hidden_size, intermediate_size; attention_dropout, dropout, return_score)

function PreNormTransformerBlock(
    act, head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    attention_dropout = nothing, dropout = nothing, return_score = false,
)
    sa = SelfAttention(head, hidden_size, head_hidden_size; dropout = attention_dropout, return_score)
    ff1 = Flux.Dense(hidden_size, intermediate_size, act)
    ff2 = Flux.Dense(intermediate_size, hidden_size)
    return TransformerBlock(
        PreNormResidual(
            DropoutLayer(sa, dropout),
            LayerNorm(hidden_size)),
        PreNormResidual(
            DropoutLayer(Flux.Chain(ff1, ff2), dropout),
            LayerNorm(hidden_size)))
end

#############################################

TransformerDecoderBlock(
    head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    dropout = nothing, attention_dropout = nothing, cross_attention_dropout = nothing,
    return_score = false, return_self_attention_score = false,
) = TransformerDecoderBlock(
    gelu, head, hidden_size, head_hidden_size, intermediate_size;
    dropout, attention_dropout, cross_attention_dropout, return_score, return_self_attention_score
)

TransformerDecoderBlock(
    act, head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    dropout = nothing, attention_dropout = nothing, cross_attention_dropout = nothing,
    return_score = false, return_self_attention_score = false,
) = PostNormTransformerDecoderBlock(
    act, head, hidden_size, head_hidden_size, intermediate_size;
    dropout, attention_dropout, cross_attention_dropout, return_score, return_self_attention_score
)

PostNormTransformerDecoderBlock(
    head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    dropout = nothing, attention_dropout = nothing, cross_attention_dropout = nothing,
    return_score = false, return_self_attention_score = false,
) = PostNormTransformerDecoderBlock(
    gelu, head, hidden_size, head_hidden_size, intermediate_size;
    dropout, attention_dropout, cross_attention_dropout, return_score, return_self_attention_score
)

function PostNormTransformerDecoderBlock(
    act, head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    dropout = nothing, attention_dropout = nothing, cross_attention_dropout = nothing,
    return_score = false, return_self_attention_score = false,
)
    sa = SelfAttention(head, hidden_size, head_hidden_size;
                       dropout = attention_dropout, causal = true, return_score = return_self_attention_score)
    ca = CrossAttention(head, hidden_size, head_hidden_size; dropout = cross_attention_dropout, return_score)
    ff1 = Flux.Dense(hidden_size, intermediate_size, act)
    ff2 = Flux.Dense(intermediate_size, hidden_size)
    return TransformerDecoderBlock(
        PostNormResidual(
            DropoutLayer(sa, dropout),
            LayerNorm(hidden_size)),
        PostNormResidual(
            DropoutLayer(ca, dropout),
            LayerNorm(hidden_size)),
        PostNormResidual(
            DropoutLayer(Flux.Chain(ff1, ff2), dropout),
            LayerNorm(hidden_size)))
end

PreNormTransformerDecoderBlock(
    head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    dropout = nothing, attention_dropout = nothing, cross_attention_dropout = nothing,
    return_score = false, return_self_attention_score = false,
) = PreNormTransformerDecoderBlock(
    gelu, head, hidden_size, head_hidden_size, intermediate_size;
    dropout, attention_dropout, cross_attention_dropout, return_score, return_self_attention_score
)

function PreNormTransformerDecoderBlock(
    act, head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    dropout = nothing, attention_dropout = nothing, cross_attention_dropout = nothing,
    return_score = false, return_self_attention_score = false,
)
    sa = SelfAttention(head, hidden_size, head_hidden_size;
                       dropout = attention_dropout, causal = true, return_score = return_self_attention_score)
    ca = CrossAttention(head, hidden_size, head_hidden_size; dropout = cross_attention_dropout, return_score)
    ff1 = Flux.Dense(hidden_size, intermediate_size, act)
    ff2 = Flux.Dense(intermediate_size, hidden_size)
    return TransformerDecoderBlock(
        PreNormResidual(
            DropoutLayer(sa, dropout),
            LayerNorm(hidden_size)),
        PreNormResidual(
            DropoutLayer(ca, dropout),
            LayerNorm(hidden_size)),
        PreNormResidual(
            DropoutLayer(Flux.Chain(ff1, ff2), dropout),
            LayerNorm(hidden_size)))
end

#############################################

struct Seq2Seq{E, D} <: LayerStruct
    encoder::E
    decoder::D
end
@functor Seq2Seq

argument_names(::Seq2Seq) = (:encoder_input, :decoder_input)

function (seq2seq::Seq2Seq)(nt::NamedTuple)
    enc = apply_on_namedtuple(seq2seq.encoder, nt.encoder_input)
    dec = apply_on_namedtuple(seq2seq.decoder, merge(nt.decoder_input, (memory = enc.hidden_state,)))
    hidden_state = dec.hidden_state
    return merge(Base.structdiff(nt, NamedTuple{(:encoder_input, :decoder_input)}),
                 (hidden_state = hidden_state, encoder_output = enc, decoder_output = dec))
end

#############################################

abstract type AbstractEmbedding end

struct Embed{F, E <: AbstractArray} <: AbstractEmbedding
    scale::F
    embeddings::E
end
@functor Embed (embeddings,)

Embed(hidden_size::Int, vocab_size::Int; scale = nothing) = Embed(scale, randn(Float32, hidden_size, vocab_size))

(embed::Embed{Nothing})(x) = NNlib.gather(embed.embeddings, x)
function (embed::Embed)(x)
    y = NNlib.gather(embed.embeddings, x)
    return y .* convert(eltype(y), embed.scale)
end

function Base.show(io::IO, embed::Embed)
    print(io, "Embed(", size(embed.embeddings, 1), ", ", size(embed.embeddings, 2))
    !isnothing(embed.scale) && print(io, ", scale = ", embed.scale)
    print(io, ')')
end

struct EmbedDecoder{E<:AbstractEmbedding}
    embed::E
end
@functor EmbedDecoder

function (e::EmbedDecoder{<:Embed})(x)
    return NeuralAttentionlib.scaled_matmul(e.embed.embeddings', x, e.embed.scale)
end
function (e::EmbedDecoder{<:Embed{Nothing}})(x)
    return NeuralAttentionlib.scaled_matmul(e.embed.embeddings', x)
end

function Base.show(io::IO, e::EmbedDecoder)
    print(io, "EmbedDecoder(")
    show(io, e.embed)
    print(io, ')')
end

struct FixedLenPositionEmbed{E <: AbstractArray} <: AbstractEmbedding
    embeddings::E
end
@functor FixedLenPositionEmbed

FixedLenPositionEmbed(hidden_size::Int, max_length::Int = 1024) =
    FixedLenPositionEmbed(randn(Float32, hidden_size, max_length))

(embed::FixedLenPositionEmbed)(x) = reshape(embed(size(x, 2)), Val(ndims(x)))
(embed::FixedLenPositionEmbed)(x::AbstractArray{<:Integer}) = NNlib.gather(embed.embeddings, x)
(embed::FixedLenPositionEmbed)(len::Int) = embed.embeddings[:, Base.OneTo(len)]

Base.show(io::IO, embed::FixedLenPositionEmbed) = (print(io, "FixedLenPositionEmbed"); print(io, size(embed.embeddings)))

struct SinCosPositionEmbed{F} <: AbstractEmbedding
    f::F
    hidden_size::Int
    normalized::Bool
end
SinCosPositionEmbed(hidden_size::Int, normalized::Bool = false) = SinCosPositionEmbed(Base.Fix1(default_position_func, hidden_size), hidden_size, normalized)
SinCosPositionEmbed(f, hidden_size::Int) = SinCosPositionEmbed(f, hidden_size, false)

@inline function default_position_func(hidden_size, i)
    j = 8 * (1 - i)
    return 1e1 ^ (j / hidden_size)
end

function sincos_position_embed(f, hidden_size, pos, indices, ::Val{normalized}) where normalized
    idx = first(Tuple(indices))
    i = (idx + 1) >> 1
    w = (pos - 1) * f(i)
    y = idx & 0x1 > 0 ? sin(w) : cos(w)
    if normalized
        return y * inv(sqrt(hidden_size >> 1))
    else
        return y
    end
end

function compute_position_embedding!(y, embed::SinCosPositionEmbed, x)
    y .= sincos_position_embed.(embed.f, embed.hidden_size, x, CartesianIndices(y), Val(embed.normalized))
    return y
end

function (embed::SinCosPositionEmbed)(x)
    len = size(x, 2)
    y = reshape(similar(x, embed.hidden_size, len), Val(ndims(x)))
    compute_position_embedding!(y, embed, Base.OneTo(len)')
    return y
end
function (embed::SinCosPositionEmbed)(x::AbstractArray{<:Integer})
    y = similar(x, Float32, embed.hidden_size, size(x)...)
    compute_position_embedding!(y, embed, reshape(x, 1, size(x)...))
    return y
end
function (embed::SinCosPositionEmbed)(len::Int)
    y = Matrix{Float32}(undef, embed.hidden_size, len)
    compute_position_embedding!(y, embed, Base.OneTo(len)')
    return y
end

ChainRulesCore.@non_differentiable (embed::SinCosPositionEmbed)(x)

function Base.show(io::IO, embed::SinCosPositionEmbed)
    print(io, "SinCosPositionEmbed(")
    if embed.f isa Base.Fix1{typeof(default_position_func)}
        print(io, "default_position_func(", embed.f.x, ')')
    else
        show(io, embed.f)
    end
    print(io, ", ", embed.hidden_size, ", normalized = ", embed.normalized, ')')
end

struct ApplyEmbed{F, E, I}
    apply::F
    embed::E
    indices::I
end
@functor ApplyEmbed (apply, embed)

ApplyEmbed(embed) = ApplyEmbed(.+, embed)
ApplyEmbed(apply, embed) = ApplyEmbed(apply, embed, identity)

function (e::ApplyEmbed)(x, indices = e.indices(x))
    embeddings = e.embed(indices)
    return e.apply(x, embeddings)
end

function Base.show(io::IO, e::ApplyEmbed)
    print(io, "ApplyEmbed(")
    show(io, e.apply)
    print(io, ", ")
    show(io, e.embed)
    if !(e.indices isa typeof(identity))
        print(io, ", ")
        show(io, e.indices)
    end
    print(io, ')')
end

struct CompositeEmbedding{T<:Tuple}  <: LayerStruct
    embeds::T
end
Functors.functor(::Type{<:CompositeEmbedding}, x) = ((embeds = getfield(x, :embeds),), y -> CompositeEmbedding(y.embeds))

argument_names(ce::CompositeEmbedding) = remove_name(argument_names(getfield(ce, :embeds)), :hidden_state)

CompositeEmbedding(args...) = CompositeEmbedding{typeof(args)}(args)
function CompositeEmbedding(; kwargs...)
    embeds = []
    for (i, name) in enumerate(keys(kwargs))
        embed = kwargs[name]
        if isone(i)
            push!(embeds, WithArg{(name,)}(embed))
        else
            if !(embed isa Tuple)
                embed = (embed,)
            end
            push!(embeds, WithOptArg{(:hidden_state,), (name,)}(ApplyEmbed(embed...)))
        end
    end
    return CompositeEmbedding(Tuple(embeds))
end

function (ce::CompositeEmbedding)(nt::NamedTuple)
    if @generated
        N = length(ce.parameters[1].parameters)
        symbols = [gensym() for _ in 1:N]
        pushfirst!(symbols, :nt)
        calls = [ :($(symbols[i+1]) = apply_on_namedtuple(ce[$i], $(symbols[i]))) for i in 1:N ]
        return Expr(:block, calls...)
    else
        applylayers(getfield(ce, :embeds), nt)
    end
end

Base.length(ce::CompositeEmbedding) = length(getfield(ce, :embeds))
Base.getindex(ce::CompositeEmbedding, i) = getfield(ce, :embeds)[i]

function Base.getproperty(ce::CompositeEmbedding, sym::Symbol)
    names = propertynames(ce)
    i = sym_in(sym, names)
    iszero(i) && error("Unknown embeddding name: $sym\nPossible names: $(propertynames(ce))")
    return getfield(ce, :embeds)[i].layer
end

Base.propertynames(ce::CompositeEmbedding) = first.(argument_names.(getfield(ce, :embeds)))

function Base.show(io::IO, ce::CompositeEmbedding)
    print(io, "CompositeEmbedding(")
    print(io, ce[1])
    for e in Base.tail(getfield(ce, :embeds))
        print(io, ", ")
        show(io, e)
    end
    print(io, ')')
end

#############################################

for T in :[
    Chain,
    LayerNorm, RMSLayerNorm, DropoutLayer, SelfAttention, CrossAttention,
    PostNormResidual, PreNormResidual, TransformerBlock, TransformerDecoderBlock,
    Transformer, Seq2Seq, CompositeEmbedding,
].args
    @eval function Base.show(io::IO, m::MIME"text/plain", x::$T)
        if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
            Flux._big_show(io, x)
        elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
            Flux._layer_show(io, x)
        else
            show(io, x)
        end
    end
end
