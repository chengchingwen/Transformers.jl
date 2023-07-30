using StructWalk
using ChainRulesCore
using NeuralAttentionlib
using NeuralAttentionlib: $, AbstractAttenOp, MultiheadQKVAttenOpWithScore, MultiheadQKVAttenOp,
    CausalMultiheadQKVAttenOp, CausalMultiheadQKVAttenOpWithScore,
    GroupedQueryAttenOp, GroupedQueryAttenOpWithScore,
    CausalGroupedQueryAttenOp, CausalGroupedQueryAttenOpWithScore,
    with_rotary_position_embedding, dot_product_score, scaled_dot_product_score,
    masked_score, normalized_score, dropout_score, weighted_sum_mixing,
    generic_multihead_qkv_attention, generic_grouped_query_attention,
    CausalMask, BatchedMask, LocalMask


WithScore(op::AbstractAttenOp) = NeuralAttentionlib.WithScore(op)
WithScore(x) = postwalk(LayerStyle, x) do xi
    xi isa AbstractAttenOp ? NeuralAttentionlib.WithScore(xi) : xi
end

set_dropout(op::MultiheadQKVAttenOp, p) = MultiheadQKVAttenOp(op.head, p)
set_dropout(op::CausalMultiheadQKVAttenOp, p) = CausalMultiheadQKVAttenOp(op.head, p)
set_dropout(op::NeuralAttentionlib.WithScore, p) = NeuralAttentionlib.WithScore(set_dropout(getfield(op, :op), p))

function apply_attention_op(op, nt::NamedTuple)
    qkv = nt.hidden_state
    ChainRulesCore.ignore_derivatives() do
        qkv isa NTuple{3, Any} ||
            error("Expect hidden_state to be a tuple of 3 arrays, but get $(typeof(qkv)).")
        nothing
    end
    q, k, v = qkv
    mask = ChainRulesCore.ignore_derivatives(()->get(nt, :attention_mask, nothing))
    a = op(q, k, v, mask)
    return return_hidden_state(nt, a)
end

# dot attention

dot_attention_score(mask, p) =
    dropout_score(p) $ normalized_score(softmax) $
    masked_score(NeuralAttentionlib.GenericMaskOp(), mask) $ dot_product_score

ChainRulesCore.@non_differentiable dot_attention_score(arg...)

function multihead_qkv_dot_attention(head, q, k, v, mask = nothing, p = nothing)
    return generic_multihead_qkv_attention(weighted_sum_mixing, dot_attention_score(mask, p), head, q, k, v)
end
function multihead_qkv_dot_attention(
    ::typeof(NeuralAttentionlib.score_returning),
    head, q, k, v, mask = nothing, p = nothing)
    return generic_multihead_qkv_attention(
        NeuralAttentionlib.score_returning(weighted_sum_mixing),
        dot_attention_score(mask, p), head, q, k, v)
end

struct MultiheadQKVDotAttenOp{F} <: AbstractAttenOp
    head::Int
    p::F
end
MultiheadQKVDotAttenOp(head) = MultiheadQKVDotAttenOp(head, nothing)
NeuralAttentionlib.get_attention_func(::MultiheadQKVDotAttenOp) = multihead_qkv_dot_attention
NeuralAttentionlib.get_attention_func_args(op::MultiheadQKVDotAttenOp, q, k, v, mask = nothing) =
    (op.head, q, k, v, BatchedMask(mask), op.p)

set_dropout(op::MultiheadQKVDotAttenOp, p) = MultiheadQKVDotAttenOp(op.head, p)

const MultiheadQKVDotAttenOpWithScore{F} = NeuralAttentionlib.WithScore{MultiheadQKVDotAttenOp{F}}

struct CausalMultiheadQKVDotAttenOp{F} <: AbstractAttenOp
    head::Int
    p::F
end
CausalMultiheadQKVDotAttenOp(head) = CausalMultiheadQKVDotAttenOp(head, nothing)
NeuralAttentionlib.get_attention_func(::CausalMultiheadQKVDotAttenOp) = multihead_qkv_dot_attention
NeuralAttentionlib.get_attention_func_args(op::CausalMultiheadQKVDotAttenOp, q, k, v, mask = nothing) =
    (op.head, q, k, v, BatchedMask(CausalMask() & mask), op.p)

set_dropout(op::CausalMultiheadQKVDotAttenOp, p) = CausalMultiheadQKVDotAttenOp(op.head, p)

const CausalMultiheadQKVDotAttenOpWithScore{F} = NeuralAttentionlib.WithScore{CausalMultiheadQKVDotAttenOp{F}}

# local

struct LocalMultiheadQKVAttenOp{F} <: AbstractAttenOp
    size::Int
    head::Int
    p::F
end
LocalMultiheadQKVAttenOp(size, head) = LocalMultiheadQKVAttenOp(size, head, nothing)
NeuralAttentionlib.get_attention_func(::LocalMultiheadQKVAttenOp) = multihead_qkv_attention
NeuralAttentionlib.get_attention_func_args(op::LocalMultiheadQKVAttenOp, q, k, v, mask = nothing) =
    (op.head, q, k, v, BatchedMask(LocalMask(op.size) & mask), op.p)

set_dropout(op::LocalMultiheadQKVAttenOp, p) = LocalMultiheadQKVAttenOp(op.size, op.head, p)

const LocalMultiheadQKVAttenOpWithScore{F} = NeuralAttentionlib.WithScore{LocalMultiheadQKVAttenOp{F}}

struct LocalCausalMultiheadQKVAttenOp{F} <: AbstractAttenOp
    size::Int
    head::Int
    p::F
end
LocalCausalMultiheadQKVAttenOp(size, head) = LocalCausalMultiheadQKVAttenOp(size, head, nothing)
NeuralAttentionlib.get_attention_func(::LocalCausalMultiheadQKVAttenOp) = multihead_qkv_attention
NeuralAttentionlib.get_attention_func_args(op::LocalCausalMultiheadQKVAttenOp, q, k, v, mask = nothing) =
    (op.head, q, k, v, BatchedMask(CausalMask() & LocalMask(op.size) & mask), op.p)

set_dropout(op::LocalCausalMultiheadQKVAttenOp, p) = LocalCausalMultiheadQKVAttenOp(op.size, op.head, p)

const LocalCausalMultiheadQKVAttenOpWithScore{F} = NeuralAttentionlib.WithScore{LocalCausalMultiheadQKVAttenOp{F}}

struct LocalMultiheadQKVDotAttenOp{F} <: AbstractAttenOp
    size::Int
    head::Int
    p::F
end
LocalMultiheadQKVDotAttenOp(size, head) = LocalMultiheadQKVDotAttenOp(size, head, nothing)
NeuralAttentionlib.get_attention_func(::LocalMultiheadQKVDotAttenOp) = multihead_qkv_dot_attention
NeuralAttentionlib.get_attention_func_args(op::LocalMultiheadQKVDotAttenOp, q, k, v, mask = nothing) =
    (op.head, q, k, v, BatchedMask(LocalMask(op.size) & mask), op.p)

set_dropout(op::LocalMultiheadQKVDotAttenOp, p) = LocalMultiheadQKVDotAttenOp(op.size, op.head, p)

const LocalMultiheadQKVDotAttenOpWithScore{F} = NeuralAttentionlib.WithScore{LocalMultiheadQKVDotAttenOp{F}}

struct LocalCausalMultiheadQKVDotAttenOp{F} <: AbstractAttenOp
    size::Int
    head::Int
    p::F
end
LocalCausalMultiheadQKVDotAttenOp(size, head) = LocalCausalMultiheadQKVDotAttenOp(size, head, nothing)
NeuralAttentionlib.get_attention_func(::LocalCausalMultiheadQKVDotAttenOp) = multihead_qkv_dot_attention
NeuralAttentionlib.get_attention_func_args(op::LocalCausalMultiheadQKVDotAttenOp, q, k, v, mask = nothing) =
    (op.head, q, k, v, BatchedMask(CausalMask() & LocalMask(op.size) & mask), op.p)

set_dropout(op::LocalCausalMultiheadQKVDotAttenOp, p) = LocalCausalMultiheadQKVDotAttenOp(op.size, op.head, p)

const LocalCausalMultiheadQKVDotAttenOpWithScore{F} = NeuralAttentionlib.WithScore{LocalCausalMultiheadQKVDotAttenOp{F}}

# RoPE

rope_attention_score(dim, mask, p) =
    dropout_score(p) $
    normalized_score(softmax) $
    masked_score(NeuralAttentionlib.GenericMaskOp(), mask) $
    scaled_dot_product_score $
    with_rotary_position_embedding(dim)

ChainRulesCore.@non_differentiable rope_attention_score(arg...)

function rope_multihead_qkv_attention(dim, head, q, k, v, mask = nothing, p = nothing)
    return generic_multihead_qkv_attention(
        weighted_sum_mixing, rope_attention_score(dim, mask, p),
        head, q, k, v)
end
function rope_multihead_qkv_attention(
    ::typeof(NeuralAttentionlib.score_returning),
    dim, head, q, k, v, mask = nothing, p = nothing
)
    return generic_multihead_qkv_attention(
        NeuralAttentionlib.score_returning(weighted_sum_mixing),
        rope_attention_score(dim, mask, p),
        head, q, k, v, position_embedding)
end

struct RoPEMultiheadQKVAttenOp{D, F} <: AbstractAttenOp
    dim::D
    head::Int
    p::F
end
RoPEMultiheadQKVAttenOp(head::Int) = RoPEMultiheadQKVAttenOp(nothing, head, nothing)
RoPEMultiheadQKVAttenOp(dim::Int, head::Int) = RoPEMultiheadQKVAttenOp(dim, head, nothing)
NeuralAttentionlib.get_attention_func(::RoPEMultiheadQKVAttenOp) = rope_multihead_qkv_attention
NeuralAttentionlib.get_attention_func_args(op::RoPEMultiheadQKVAttenOp, q, k, v, mask = nothing) =
    (op.dim, op.head, q, k, v, BatchedMask(mask), op.p)

set_dropout(op::RoPEMultiheadQKVAttenOp, p) = RoPEMultiheadQKVAttenOp(op.dim, op.head, p)

const RoPEMultiheadQKVAttenOpWithScore{D, F} = NeuralAttentionlib.WithScore{RoPEMultiheadQKVAttenOp{D, F}}

struct CausalRoPEMultiheadQKVAttenOp{D, F} <: AbstractAttenOp
    dim::D
    head::Int
    p::F
end
CausalRoPEMultiheadQKVAttenOp(head::Int) = CausalRoPEMultiheadQKVAttenOp(nothing, head, nothing)
CausalRoPEMultiheadQKVAttenOp(dim::Int, head::Int) = CausalRoPEMultiheadQKVAttenOp(dim, head, nothing)
NeuralAttentionlib.get_attention_func(::CausalRoPEMultiheadQKVAttenOp) = rope_multihead_qkv_attention
NeuralAttentionlib.get_attention_func_args(op::CausalRoPEMultiheadQKVAttenOp, q, k, v, mask = nothing) =
    (op.dim, op.head, q, k, v, BatchedMask(CausalMask() & mask), op.p)

set_dropout(op::CausalRoPEMultiheadQKVAttenOp, p) = CausalRoPEMultiheadQKVAttenOp(op.dim, op.head, p)

const CausalRoPEMultiheadQKVAttenOpWithScore{D, F} = NeuralAttentionlib.WithScore{CausalRoPEMultiheadQKVAttenOp{D, F}}

# layer api

for op in :[
    MultiheadQKVAttenOp, MultiheadQKVAttenOpWithScore,
    CausalMultiheadQKVAttenOp, CausalMultiheadQKVAttenOpWithScore,
    MultiheadQKVDotAttenOp, MultiheadQKVDotAttenOpWithScore,
    CausalMultiheadQKVDotAttenOp, CausalMultiheadQKVDotAttenOpWithScore,
    LocalMultiheadQKVAttenOp, LocalMultiheadQKVAttenOpWithScore,
    LocalCausalMultiheadQKVAttenOp, LocalCausalMultiheadQKVAttenOpWithScore,
    LocalMultiheadQKVDotAttenOp, LocalMultiheadQKVDotAttenOpWithScore,
    LocalCausalMultiheadQKVDotAttenOp, LocalCausalMultiheadQKVDotAttenOpWithScore,
    RoPEMultiheadQKVAttenOp, RoPEMultiheadQKVAttenOpWithScore,
    CausalRoPEMultiheadQKVAttenOp, CausalRoPEMultiheadQKVAttenOpWithScore,
    GroupedQueryAttenOp, GroupedQueryAttenOpWithScore,
    CausalGroupedQueryAttenOp, CausalGroupedQueryAttenOpWithScore,
].args
    @eval begin
        argument_names(::$op) = (:hidden_state, :attention_mask)
        apply_on_namedtuple(op::$op, nt::NamedTuple) = apply_attention_op(op, nt)
    end
end
