import ..Layers
using NeuralAttentionlib
using NeuralAttentionlib: $, AbstractAttenOp,
    with_rotary_position_embedding, scaled_dot_product_score,
    masked_score, normalized_score, dropout_score, weighted_sum_mixing,
    generic_grouped_query_attention, CausalMask, BatchedMask
using Static
using ChainRulesCore


llama_rope_attention_score(dim, mask, p) =
    dropout_score(p) $
    normalized_score(softmax) $
    masked_score(NeuralAttentionlib.GenericMaskOp(), mask) $
    scaled_dot_product_score $
    (with_rotary_position_embedding(dim) ∘ gptneox_reorder(dim))

ChainRulesCore.@non_differentiable llama_rope_attention_score(args...)

function llama_rope_grouped_query_attention(dim, head, group, q, k, v, mask = nothing, p = nothing)
    return generic_grouped_query_attention(
        weighted_sum_mixing, llama_rope_attention_score(dim, mask, p),
        head, group, q, k, v)
end
function llama_rope_grouped_query_attention(
    ::typeof(NeuralAttentionlib.score_returning),
    dim, head, group, q, k, v, mask = nothing, p = nothing
)
    return generic_grouped_query_attention(
        NeuralAttentionlib.score_returning(weighted_sum_mixing),
        llama_rope_attention_score(dim, mask, p),
        head, group, q, k, v)
end

struct CausalLlamaRoPEGroupedQueryAttenOp{D, P} <: AbstractAttenOp
    dim::D
    head::Int
    group::Int
    p::P
end
CausalLlamaRoPEGroupedQueryAttenOp(head::Int, group::Int) =
    CausalLlamaRoPEGroupedQueryAttenOp(nothing, head, group, nothing)
CausalLlamaRoPEGroupedQueryAttenOp(dim::Int, head::Int, group::Int) =
    CausalLlamaRoPEGroupedQueryAttenOp(dim, head, group, nothing)

NeuralAttentionlib.get_attention_func(::CausalLlamaRoPEGroupedQueryAttenOp) = llama_rope_grouped_query_attention
NeuralAttentionlib.get_attention_func_args(op::CausalLlamaRoPEGroupedQueryAttenOp, q, k, v, mask = nothing) =
    (op.dim, op.head, op.group, q, k, v, BatchedMask(CausalMask() & mask), op.p)

Layers.set_dropout(op::CausalLlamaRoPEGroupedQueryAttenOp, p) = CausalLlamaRoPEGroupedQueryAttenOp(op.dim, op.head, op.group, p)

const CausalLlamaRoPEGroupedQueryAttenOpWithScore{D, P} = NeuralAttentionlib.WithScore{CausalLlamaRoPEGroupedQueryAttenOp{D, P}}

Layers.argument_names(::CausalLlamaRoPEGroupedQueryAttenOp) = (:hidden_state, :attention_mask)
Layers.apply_on_namedtuple(op::CausalLlamaRoPEGroupedQueryAttenOp, nt::NamedTuple) = Layers.apply_attention_op(op, nt)
