using Flux
using Functors
using Static

using NeuralAttentionlib
using NeuralAttentionlib: $, AbstractAttenOp, multihead_qkv_attention, score_returning

@inline get_hidden_state(x::NamedTuple) = haskey(x, :hidden_state) ? x.hidden_state : x
@inline get_hidden_state(x) = x

@inline update_hidden_state(_, y) = y
@inline update_hidden_state(x::NamedTuple, y) = merge(x, (hidden_state = y,))

@inline function apply_on_hidden_state(f, nt, args...)
    x = get_hidden_state(nt)
    y = f(x, args...)
    return update_hidden_state(nt, y)
end

@inline function apply_on_hidden_state(::typeof(+), nt, args...)
    x = get_hidden_state(nt)
    x_args = map(get_hidden_state, args)
    y = +(x, x_args...)
    return update_hidden_state(nt, y)
end

struct MultiheadQKVAttenOp <: AbstractAttenOp
    head::Int
    p::Union{Float64, Nothing}
end
MultiheadQKVAttenOp(head) = MultiheadQKVAttenOp(head, nothing)

function (op::MultiheadQKVAttenOp)(q, k, v, mask = nothing)
    return isnothing(op.p) ?
        multihead_qkv_attention(op.head, q, k, v, mask) :
        multihead_qkv_attention(op.head, q, k, v, mask, op.p)
end

struct MultiheadQKVAttenOpWithScore <: AbstractAttenOp
    head::Int
    p::Union{Float64, Nothing}
end
MultiheadQKVAttenOpWithScore(head) = MultiheadQKVAttenOpWithScore(head, nothing)

function (op::MultiheadQKVAttenOpWithScore)(q, k, v, mask = nothing)
    return isnothing(op.p) ?
        multihead_qkv_attention(score_returning, op.head, q, k, v, mask) :
        multihead_qkv_attention(score_returning, op.head, q, k, v, mask, op.p)
end

struct NSplit{N, L}
    n::N
    layer::L
end
NSplit(n::Integer, layer) = NSplit(static(n), layer)

@functor NSplit

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

struct DropoutLayer{L, P}
    layer::L
    p::P
end

@functor DropoutLayer

(drop::DropoutLayer)(x) = apply_on_hidden_state(drop, x)
(drop::DropoutLayer)(x, mask) = apply_on_hidden_state(drop, x, mask)

function apply_on_hidden_state(drop::DropoutLayer, nt, args...)
    y = apply_on_hidden_state(drop.layer, nt, args...)
    return isnothing(drop.p) ? y : apply_on_hidden_state(Flux.dropout, y, drop.p)
end

struct TransformerBlock{A, F}
    attention::A
    feedforward::F
end

@functor TransformerBlock

(b::TransformerBlock)(x) = b.feedforward(b.attention(x))
(b::TransformerBlock)(x, mask) = b.feedforward(b.attention(x, mask))

struct PreNormResidual{L, N}
    layer::L
    norm::N
end

@functor PreNormResidual

function (prenr::PreNormResidual)(x)
    norm = apply_on_hidden_state(prenr.norm, x)
    y = apply_on_hidden_state(prenr.layer, norm)
    return apply_on_hidden_state(+, y, x)
end
function (prenr::PreNormResidual)(x, mask)
    norm = apply_on_hidden_state(prenr.norm, x)
    y = apply_on_hidden_state(prenr.layer, norm, mask)
    return apply_on_hidden_state(+, y, x)
end

struct PostNormResidual{L, N}
    layer::L
    norm::N
end

@functor PostNormResidual

function (postnr::PostNormResidual)(x)
    y = apply_on_hidden_state(postnr.layer, x)
    r = apply_on_hidden_state(+, y, x)
    return apply_on_hidden_state(postnr.norm, r)
end
function (postnr::PostNormResidual)(x, mask)
    y = apply_on_hidden_state(postnr.layer, x, mask)
    r = apply_on_hidden_state(+, y, x)
    return apply_on_hidden_state(postnr.norm, r)
end

struct SelfAttention{A, QKV, O}
    attention_op::A
    qkv_proj::NSplit{StaticInt{3}, QKV}
    o_proj::O
end

@functor SelfAttention

function (sa::SelfAttention)(x, mask = nothing)
    a = sa.attention_op(sa.qkv_proj(x)..., mask)
    return apply_on_hidden_state(sa.o_proj, a)
end

function apply_on_hidden_state(sa::SelfAttention, nt, args...)
    x = get_hidden_state(nt)
    y = sa(x, args...)
    if nt isa NamedTuple
        if y isa NamedTuple
            return merge(nt, y)
        else
            return update_hidden_state(nt, y)
        end
    else
        return y
    end
end

struct Transformer{T <: Tuple{Vararg{TransformerBlock}}, F}
    blocks::T
    f::F
end
Transformer(blocks::Tuple{Vararg{TransformerBlock}}) = Transformer(blocks, nothing)
Transformer(blocks::TransformerBlock...) = Transformer(blocks)

@functor Transformer

(t::Transformer)(x) = applyblocks(t.blocks, t.f, x, nothing)
(t::Transformer)(x, mask) = applyblocks(t.blocks, t.f, x, mask)

function _block_call(symbols, i, has_mask, has_f)
    call = :(blocks[$i]($(symbols[i])))
    has_mask && push!(call.args, :mask)
    if has_f
        call = :(f($(symbols[i]), $call))
    end
    line = :($(symbols[i+1]) = $call)
    return line
end

function applyblocks(blocks::Tuple{Vararg{TransformerBlock, N}}, f, x, mask) where N
    if @generated
        symbols = vcat(:x, [gensym() for _ in 1:N])
        has_mask = !(mask <: Nothing)
        has_f = !(f <: Nothing)
        calls = [ _block_call(symbols, i, has_mask, has_f) for i in 1:N ]
        return Expr(:block, calls...)
    else
        if isnothing(f)
            return isnothing(mask) ?
                foldl((y, blk)-> blk(y), blocks; init=x) :
                foldl((y, blk)-> blk(y, mask), blocks; init=x)
        else
            return isnothing(mask) ?
                foldl((y, blk)-> f(y, blk(y)), blocks; init=x) :
                foldl((y, blk)-> f(y, blk(y, mask)), blocks; init=x)
        end
    end
end

Base.getindex(t::Transformer, i::Integer) = t.blocks[i]
Base.getindex(t::Transformer, r::AbstractVector) = Transformer(t.blocks[r])

remove_outputs(x) = x
remove_outputs(x::NamedTuple) = Base.structdiff(x, NamedTuple{(:outputs,)})

@inline _output_tuple(x::NamedTuple, y::NamedTuple{(:hidden_state,)}) = _output_tuple(x, y.hidden_state)
@inline function _output_tuple(x::NamedTuple, new_output)
    if haskey(x, :outputs)
        outputs = x.outputs
        outputs = (outputs..., new_output)
    else
        outputs = (new_output,)
    end
    return outputs
end

@inline collect_outputs(x, output) = collect_outputs((hidden_state = x,), output)#(hidden_state = x, outputs = (output,))
@inline function collect_outputs(x::NamedTuple, output)
    hidden_state = get_hidden_state(output)
    new_output = remove_outputs(output)
    outputs = _output_tuple(x, new_output)
    return merge(x, (hidden_state = hidden_state, outputs = outputs,))
end

TransformerBlock(
    head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int,
    attention_dropout = nothing, dropout = nothing;
    return_score = false,
) = TransformerBlock(gelu, head, hidden_size, head_hidden_size, intermediate_size, attention_dropout, dropout; return_score)

function TransformerBlock(
    act, head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int,
    attention_dropout = nothing, dropout = nothing;
    return_score = false,
)
    atten_op = return_score ?
        MultiheadQKVAttenOpWithScore(head, attention_dropout) :
        MultiheadQKVAttenOp(head, attention_dropout)
    qkv_proj = Dense(hidden_size, 3head * head_hidden_size)
    o_proj = Dense(head * head_hidden_size, hidden_size)
    sa = SelfAttention(atten_op, NSplit(static(3), qkv_proj), o_proj)
    a = DropoutLayer(sa, dropout)
    ff1 = Dense(hidden_size, intermediate_size, act)
    ff2 = Dense(intermediate_size, hidden_size)
    ff = DropoutLayer(Chain(ff1, ff2), dropout)
    ln1 = LayerNorm(hidden_size)
    ln2 = LayerNorm(hidden_size)
    return TransformerBlock(PostNormResidual(a, ln1), PostNormResidual(ff, ln2))
end

Transformer(
    n::Int, head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int,
    attention_dropout = nothing, dropout = nothing;
    collect_outputs = false, return_score = false,
) = Transformer(n, gelu, head, hidden_size, head_hidden_size, intermediate_size, attention_dropout, dropout;
                collect_outputs, return_score)

function Transformer(
    n::Int, act, head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int,
    attention_dropout = nothing, dropout = nothing;
    collect_outputs = false, return_score = false,
)
    collect_outputs = collect_outputs || return_score # return_score always collect_outputs
    return Transformer(
        ntuple(i->TransformerBlock(
            act, head, hidden_size, head_hidden_size, intermediate_size,
            attention_dropout, dropout;
            return_score
        ), n),
        collect_outputs ? (@__MODULE__).collect_outputs : nothing
    )
end
