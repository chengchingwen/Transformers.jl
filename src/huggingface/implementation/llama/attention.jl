import ..Layers
using NeuralAttentionlib
using NeuralAttentionlib: $, AbstractAttenOp,
    with_rotary_position_embedding, scaled_dot_product_score,
    masked_score, normalized_score, dropout_score, weighted_sum_mixing,
    generic_grouped_query_attention, CausalMask, BatchedMask
using Static
using ChainRulesCore

llama_rope_attention_score(base, dim, mask, p) =
    dropout_score(p) $
    normalized_score(softmax) $
    masked_score(NeuralAttentionlib.GenericMaskOp(), mask) $
    scaled_dot_product_score $
    (with_rotary_position_embedding(base_position_func(base, dim), dim) ∘ gptneox_reorder(dim))

ChainRulesCore.@non_differentiable llama_rope_attention_score(args...)

function llama_rope_grouped_query_attention(base, dim, head, group, q, k, v, mask = nothing, p = nothing)
    return generic_grouped_query_attention(
        weighted_sum_mixing, llama_rope_attention_score(base, dim, mask, p),
        head, group, q, k, v)
end
function llama_rope_grouped_query_attention(
    ::typeof(NeuralAttentionlib.score_returning),
    base, dim, head, group, q, k, v, mask = nothing, p = nothing
)
    return generic_grouped_query_attention(
        NeuralAttentionlib.score_returning(weighted_sum_mixing),
        llama_rope_attention_score(base, dim, mask, p),
        head, group, q, k, v)
end

struct CausalLlamaRoPEGroupedQueryAttenOp{F, D, P} <: AbstractAttenOp
    base::F
    dim::D
    head::Int
    group::Int
    p::P
end
CausalLlamaRoPEGroupedQueryAttenOp(head::Int, group::Int) =
    CausalLlamaRoPEGroupedQueryAttenOp(1e4, head, group)
CausalLlamaRoPEGroupedQueryAttenOp(dim::Int, head::Int, group::Int) =
    CausalLlamaRoPEGroupedQueryAttenOp(1e4, dim, head, group)
CausalLlamaRoPEGroupedQueryAttenOp(base, head::Int, group::Int) =
    CausalLlamaRoPEGroupedQueryAttenOp(base, nothing, head, group, nothing)
CausalLlamaRoPEGroupedQueryAttenOp(base, dim::Int, head::Int, group::Int) =
    CausalLlamaRoPEGroupedQueryAttenOp(base, dim, head, group, nothing)

NeuralAttentionlib.get_attention_func(::CausalLlamaRoPEGroupedQueryAttenOp) = llama_rope_grouped_query_attention
NeuralAttentionlib.get_attention_func_args(op::CausalLlamaRoPEGroupedQueryAttenOp, q, k, v, mask = nothing) =
    (op.base, op.dim, op.head, op.group, q, k, v, BatchedMask(CausalMask() & mask), op.p)

Layers.set_dropout(op::CausalLlamaRoPEGroupedQueryAttenOp, p) = CausalLlamaRoPEGroupedQueryAttenOp(op.base, op.dim, op.head, op.group, p)

const CausalLlamaRoPEGroupedQueryAttenOpWithScore{F, D, P} = NeuralAttentionlib.WithScore{CausalLlamaRoPEGroupedQueryAttenOp{F, D, P}}

Layers.argument_names(::CausalLlamaRoPEGroupedQueryAttenOp) = (:hidden_state, :attention_mask)
Layers.apply_on_namedtuple(op::CausalLlamaRoPEGroupedQueryAttenOp, nt::NamedTuple) = Layers.apply_attention_op(op, nt)
