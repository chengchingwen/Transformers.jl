using ChainRulesCore
using NeuralAttentionlib
using NeuralAttentionlib: $, AbstractAttenOp, WithScore, t5_causal_bucketed_position_id, t5_bucketed_position_id,
    dot_product_score, masked_score, normalized_score, biased_score, dropout_score, weighted_sum_mixing,
    generic_multihead_qkv_attention


function _t5_attention(mask, p, bias = nothing)
    scoref = normalized_score(softmax) $ masked_score(NeuralAttentionlib.GenericMaskOp(), mask)
    scoref = isnothing(p) ? scoref : dropout_score(p) $ scoref
    scoref = isnothing(bias) ? scoref : (scoref $ biased_score(bias))
    return scoref
end

function t5rpe_attention_score(
    n_bucket::Integer, max_distance::Int, causal::Bool,
    q::AbstractArray, k::AbstractArray, position_embedding::AbstractArray, mask, p
)
    score = dot_product_score(q, k)
    rpe_id = causal ?
        t5_causal_bucketed_position_id(n_bucket, max_distance) :
        t5_bucketed_position_id(n_bucket, max_distance)
    position_bias = NeuralAttentionlib.get_scalar_relative_position_embeddings(rpe_id, position_embedding, score)
    scoref = _t5_attention(mask, p, position_bias)
    attention_score = scoref(identity, score)
    return (; attention_score, position_bias)
end

function ChainRulesCore.rrule(
    config::RuleConfig, ::typeof(t5rpe_attention_score),
    n_bucket::Integer, max_distance::Int, causal::Bool,
    q::AbstractArray, k::AbstractArray, position_embedding::AbstractArray, mask, p
)
    score, score_pullback = rrule(config, dot_product_score, q, k)
    rpe_id = causal ?
        t5_causal_bucketed_position_id(n_bucket, max_distance) :
        t5_bucketed_position_id(n_bucket, max_distance)
    position_bias, rpe_pullback = rrule(
        config, NeuralAttentionlib.get_scalar_relative_position_embeddings, rpe_id, position_embedding, score)
    scoref = _t5_attention(mask, p, position_bias)
    attention_score, attention_pullback = rrule(config, scoref, identity, score)
    function t5rpe_attention_score_pullback(Ȳ)
        ∂pf, _, ∂score = attention_pullback(Ȳ.attention_score)
        if ∂pf isa NoTangent
            ∂bias = Ȳ.position_bias
        else
            ∂bias = unthunk(Ȳ.position_bias) + unthunk(∂pf.arg[end])
        end
        ∂emb = rpe_pullback(∂bias)[3]
        _, ∂q, ∂k = score_pullback(∂score)
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), ∂q, ∂k, ∂emb, NoTangent(), NoTangent())
    end
    return (; attention_score, position_bias), t5rpe_attention_score_pullback
end

t5mixing(nt, v) = NamedTuple{(:hidden_state, :position_bias)}(t5mixing_ws(nt, v))
function t5mixing_ws(nt, v)
    s = nt.attention_score
    y = weighted_sum_mixing(s, v)
    return (hidden_state = y, attention_score = NeuralAttentionlib.unwrap_collapse(s), position_bias = nt.position_bias)
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(t5mixing), nt, v)
    nt2, mixing_pullback = rrule(config, t5mixing_ws, nt, v)
    return NamedTuple{(:hidden_state, :position_bias)}(nt2), mixing_pullback
end
function ChainRulesCore.rrule(config::RuleConfig, ::typeof(t5mixing_ws), nt, v)
    s = nt.attention_score
    bias = nt.position_bias
    y, mixing_pullback = rrule(config, weighted_sum_mixing, s, v)
    s′, unwrap_pullback = rrule(config, NeuralAttentionlib.unwrap_collapse, s)
    function t5mixing_pullback(Ȳ)
        _, ∂s1 = unwrap_pullback(Ȳ.attention_score)
        _, ∂s2, ∂v = mixing_pullback(Ȳ.hidden_state)
        ∂s = unthunk(∂s1) + unthunk(∂s2)
        ∂nt = (attention_score = ∂s, position_bias = Ȳ.position_bias)
        return (NoTangent(), Tangent{Any, typeof(∂nt)}(∂nt), ∂v)
    end
    return (; hidden_state = y, attention_score = s′, position_bias = bias), t5mixing_pullback
end

function t5rpe_multihead_qkv_attention(
    head::Integer, n_bucket::Integer, max_distance::Int, causal::Bool,
    q::AbstractArray, k::AbstractArray, v::AbstractArray, position_embedding::AbstractArray,
    mask = nothing, p = nothing
)
    return generic_multihead_qkv_attention(
        t5mixing, t5rpe_attention_score $ n_bucket $ max_distance $ causal,
        head, q, k, v, position_embedding, mask, p
    )
end

function t5rpe_multihead_qkv_attention(
    ::typeof(NeuralAttentionlib.score_returning),
    head::Integer, n_bucket::Integer, max_distance::Int, causal::Bool,
    q::AbstractArray, k::AbstractArray, v::AbstractArray, position_embedding::AbstractArray,
    mask = nothing, p = nothing
)
    return generic_multihead_qkv_attention(
        t5mixing_ws, t5rpe_attention_score $ n_bucket $ max_distance $ causal,
        head, q, k, v, position_embedding, mask, p
    )
end

function t5_multihead_qkv_attention(
    head::Integer, q::AbstractArray, k::AbstractArray, v::AbstractArray, bias::Union{Nothing, AbstractArray},
    mask = nothing, p = nothing
)
    return generic_multihead_qkv_attention(
        weighted_sum_mixing,
        _t5_attention(mask, p, bias) $
        dot_product_score,
        head, q, k, v)
end

function t5_multihead_qkv_attention(
    ::typeof(NeuralAttentionlib.score_returning),
    head::Integer, q::AbstractArray, k::AbstractArray, v::AbstractArray, bias::Union{Nothing, AbstractArray},
    mask = nothing, p = nothing
)
    return generic_multihead_qkv_attention(
        NeuralAttentionlib.score_returning(weighted_sum_mixing),
        _t5_attention(mask, p, bias) $
        dot_product_score,
        head, q, k, v)
end

struct T5RPEMultiheadQKVAttenOp{F, E} <: AbstractAttenOp
    head::Int
    n_bucket::Int
    max_distance::Int
    position_embedding::E
    p::F
end
@functor T5RPEMultiheadQKVAttenOp (position_embedding,)

T5RPEMultiheadQKVAttenOp(head, n_bucket, max_distance, position_embedding) =
    T5RPEMultiheadQKVAttenOp(head, n_bucket, max_distance, position_embedding, nothing)

NeuralAttentionlib.get_attention_func(::T5RPEMultiheadQKVAttenOp) = t5rpe_multihead_qkv_attention
NeuralAttentionlib.get_attention_func_args(op::T5RPEMultiheadQKVAttenOp, q, k, v, mask = nothing) =
    (op.head, op.n_bucket, op.max_distance, false, q, k, v, op.position_embedding,
     NeuralAttentionlib.BatchedMask(mask), op.p)

function Base.show(io::IO, op::T5RPEMultiheadQKVAttenOp)
    print(io, "T5RPEMultiheadQKVAttenOp(head = ", op.head, ", n_bucket = ", op.n_bucket,
          ", max_distance = ", op.max_distance, ", position_embedding = ", reverse(size(op.position_embedding)),
          ", p = ", op.p, ')')
end

Layers.set_dropout(op::T5RPEMultiheadQKVAttenOp, p) =
    T5RPEMultiheadQKVAttenOp(op.head, op.n_bucket, op.max_distance, op.position_embedding, p)

const T5RPEMultiheadQKVAttenOpWithScore{F, E} = WithScore{T5RPEMultiheadQKVAttenOp{F, E}}
function Functors.functor(::Type{<:T5RPEMultiheadQKVAttenOpWithScore}, op)
    head, n_bucket, max_distance, p = op.head, op.n_bucket, op.max_distance, op.p
    return ((position_embedding = op.position_embedding,),
            y -> T5RPEMultiheadQKVAttenOpWithScore(head, n_bucket, max_distance, y.position_embedding, p))
end

struct T5RPECausalMultiheadQKVAttenOp{F, E} <: AbstractAttenOp
    head::Int
    n_bucket::Int
    max_distance::Int
    position_embedding::E
    p::F
end
@functor T5RPECausalMultiheadQKVAttenOp (position_embedding,)

T5RPECausalMultiheadQKVAttenOp(head, n_bucket, max_distance, position_embedding) =
    T5RPECausalMultiheadQKVAttenOp(head, n_bucket, max_distance, position_embedding, nothing)

NeuralAttentionlib.get_attention_func(::T5RPECausalMultiheadQKVAttenOp) = t5rpe_multihead_qkv_attention
NeuralAttentionlib.get_attention_func_args(op::T5RPECausalMultiheadQKVAttenOp, q, k, v, mask = nothing) =
    (op.head, op.n_bucket, op.max_distance, true, q, k, v, op.position_embedding,
     NeuralAttentionlib.BatchedMask(mask & NeuralAttentionlib.CausalMask()), op.p)

function Base.show(io::IO, op::T5RPECausalMultiheadQKVAttenOp)
    print(io, "T5RPECausalMultiheadQKVAttenOp(head = ", op.head, ", n_bucket = ", op.n_bucket,
          ", max_distance = ", op.max_distance, ", position_embedding = ", reverse(size(op.position_embedding)),
          ", p = ", op.p, ')')
end

Layers.set_dropout(op::T5RPECausalMultiheadQKVAttenOp, p) =
    T5RPECausalMultiheadQKVAttenOp(op.head, op.n_bucket, op.max_distance, op.position_embedding, p)

const T5RPECausalMultiheadQKVAttenOpWithScore{F, E} = WithScore{T5RPECausalMultiheadQKVAttenOp{F, E}}
function Functors.functor(::Type{<:T5RPECausalMultiheadQKVAttenOpWithScore}, op)
    head, n_bucket, max_distance, p = op.head, op.n_bucket, op.max_distance, op.p
    return ((position_embedding = op.position_embedding,),
            y -> T5RPECausalMultiheadQKVAttenOpWithScore(head, n_bucket, max_distance, y.position_embedding, p))
end

struct T5BiasedMultiheadQKVAttenOp{F} <: AbstractAttenOp
    head::Int
    p::F
end
T5BiasedMultiheadQKVAttenOp(head) = T5BiasedMultiheadQKVAttenOp(head, nothing)
NeuralAttentionlib.get_attention_func(::T5BiasedMultiheadQKVAttenOp) = t5_multihead_qkv_attention
NeuralAttentionlib.get_attention_func_args(op::T5BiasedMultiheadQKVAttenOp, q, k, v, bias, mask = nothing) =
    (op.head, q, k, v, bias, NeuralAttentionlib.BatchedMask(mask), op.p)

Layers.set_dropout(op::T5BiasedMultiheadQKVAttenOp, p) = T5BiasedMultiheadQKVAttenOp(op.head, p)

const T5BiasedMultiheadQKVAttenOpWithScore{F} = WithScore{T5BiasedMultiheadQKVAttenOp{F}}

struct T5BiasedCausalMultiheadQKVAttenOp{F} <: AbstractAttenOp
    head::Int
    p::F
end
T5BiasedCausalMultiheadQKVAttenOp(head) = T5BiasedCausalMultiheadQKVAttenOp(head, nothing)
NeuralAttentionlib.get_attention_func(::T5BiasedCausalMultiheadQKVAttenOp) = t5_multihead_qkv_attention
NeuralAttentionlib.get_attention_func_args(op::T5BiasedCausalMultiheadQKVAttenOp, q, k, v, bias, mask = nothing) =
    (op.head, q, k, v, bias, NeuralAttentionlib.BatchedMask(mask & NeuralAttentionlib.CausalMask()), op.p)

Layers.set_dropout(op::T5BiasedCausalMultiheadQKVAttenOp, p) = T5BiasedCausalMultiheadQKVAttenOp(op.head, p)

const T5BiasedCausalMultiheadQKVAttenOpWithScore{F} = WithScore{T5BiasedCausalMultiheadQKVAttenOp{F}}

struct T5MultiheadQKVAttenOp{F} <: AbstractAttenOp
    head::Int
    p::F
end
T5MultiheadQKVAttenOp(head) = T5MultiheadQKVAttenOp(head, nothing)
NeuralAttentionlib.get_attention_func(::T5MultiheadQKVAttenOp) = t5_multihead_qkv_attention
NeuralAttentionlib.get_attention_func_args(op::T5MultiheadQKVAttenOp, q, k, v, mask = nothing) =
    (op.head, q, k, v, nothing, NeuralAttentionlib.BatchedMask(mask), op.p)

Layers.set_dropout(op::T5MultiheadQKVAttenOp, p) = T5MultiheadQKVAttenOp(op.head, p)

const T5MultiheadQKVAttenOpWithScore{F} = WithScore{T5MultiheadQKVAttenOp{F}}

Layers.argument_names(
    ::Union{T5RPEMultiheadQKVAttenOpWithScore, T5RPEMultiheadQKVAttenOp,
            T5RPECausalMultiheadQKVAttenOpWithScore, T5RPECausalMultiheadQKVAttenOp,
            T5MultiheadQKVAttenOpWithScore, T5MultiheadQKVAttenOp}
) = (:hidden_state, :attention_mask)
Layers.argument_names(
    ::Union{T5BiasedMultiheadQKVAttenOpWithScore, T5BiasedMultiheadQKVAttenOp,
            T5BiasedCausalMultiheadQKVAttenOpWithScore, T5BiasedCausalMultiheadQKVAttenOp}
) = (:hidden_state, :attention_mask, :position_bias)

function Layers.apply_on_namedtuple(
    op::Union{T5RPEMultiheadQKVAttenOpWithScore, T5RPEMultiheadQKVAttenOp,
              T5RPECausalMultiheadQKVAttenOpWithScore, T5RPECausalMultiheadQKVAttenOp,
              T5MultiheadQKVAttenOpWithScore, T5MultiheadQKVAttenOp},
    nt::NamedTuple
)
    return Layers.apply_attention_op(op, nt)
end

function Layers.apply_on_namedtuple(
    op::Union{T5BiasedMultiheadQKVAttenOpWithScore, T5BiasedMultiheadQKVAttenOp,
              T5BiasedCausalMultiheadQKVAttenOpWithScore, T5BiasedCausalMultiheadQKVAttenOp},
    nt::NamedTuple
)
    qkv = nt.hidden_state
    qkv isa NTuple{3, Any} ||
        error("Expect hidden_state to be a tuple of 3 arrays, but get $(typeof(qkv)).")
    mask = get(nt, :attention_mask, nothing)
    bias = nt.position_bias
    a = op(qkv..., bias, mask)
    return Layers.return_hidden_state(nt, a)
end
