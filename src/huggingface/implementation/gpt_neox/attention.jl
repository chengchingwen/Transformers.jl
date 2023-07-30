import ..Layers
using NeuralAttentionlib
using NeuralAttentionlib: $, AbstractAttenOp,
    with_rotary_position_embedding, scaled_dot_product_score,
    masked_score, normalized_score, dropout_score, weighted_sum_mixing,
    generic_multihead_qkv_attention, CausalMask, BatchedMask
using Static
using ChainRulesCore

base_position_func(hidden_size) = base_position_func(1e4, hidden_size)
base_position_func(base, hidden_size) = base_position_func $ base $ hidden_size
@inline function base_position_func(base, hidden_size, i)
    j = 2(1 - i)
    return base ^ (j / hidden_size)
end

function _reorder(r, c, dim, half)
    t = Tuple(c)
    i = first(t)
    is = Base.tail(t)
    if i <= dim
        _i, h = fldmod1(i, 0x2)
        j = isone(h) ? _i : _i + half
        x = @inbounds r[j, is...]
        return x
    else
        x = @inbounds r[c]
    end
end

function ∇_reorder(r, c, dim, half)
    t = Tuple(c)
    i = first(t)
    is = Base.tail(t)
    if i <= dim
        if i <= half
            j = i << 0x1 - 0x1
        else
            j = (i - half) << 0x1
        end
        x = @inbounds r[j, is...]
        return x
    else
        x = @inbounds r[c]
    end
end

gptneox_reorder(dim) = Base.Fix1(gptneox_reorder, dim)
function gptneox_reorder(dim, x)
    dim = isnothing(dim) ? size(x, 1) : dim
    @assert iseven(dim) "rotary position embedding require the feature dim is even."
    y = similar(x)
    y .= _reorder.(Ref(x), CartesianIndices(x), dim, dim >> 1)
    return y
end

function ∇gptneox_reorder(dim, x)
    dim = isnothing(dim) ? size(x, 1) : dim
    @assert iseven(dim) "rotary position embedding require the feature dim is even."
    y = similar(x)
    y .= ∇_reorder.(Ref(x), CartesianIndices(x), dim, dim >> 1)
    return y
end

function ChainRulesCore.rrule(confg::RuleConfig, ::typeof(gptneox_reorder), dim)
    pullback(_) = (NoTangent(), NoTangent())
    return gptneox_reorder(dim), pullback
end
function ChainRulesCore.rrule(confg::RuleConfig, ::typeof(gptneox_reorder), dim, x)
    function reorder_pullback(Ybar)
        Ȳ = unthunk(Ybar)
        ∂x = ∇gptneox_reorder(dim, Ȳ)
        return (NoTangent(), NoTangent(), ∂x)
    end
    return gptneox_reorder(dim, x), reorder_pullback
end

gptneox_rope_attention_score(base, dim, mask, p) =
    dropout_score(p) $
    normalized_score(softmax) $
    masked_score(NeuralAttentionlib.GenericMaskOp(), mask) $
    scaled_dot_product_score $
    (with_rotary_position_embedding(base_position_func(base, dim), dim) ∘ gptneox_reorder(dim))

ChainRulesCore.@non_differentiable gptneox_rope_attention_score(arg...)

function gptneox_rope_multihead_qkv_attention(base, dim, head, q, k, v, mask = nothing, p = nothing)
    return generic_multihead_qkv_attention(
        weighted_sum_mixing, gptneox_rope_attention_score(base, dim, mask, p),
        head, q, k, v)
end
function gptneox_rope_multihead_qkv_attention(
    ::typeof(NeuralAttentionlib.score_returning),
    base, dim, head, q, k, v, mask = nothing, p = nothing
)
    return generic_multihead_qkv_attention(
        NeuralAttentionlib.score_returning(weighted_sum_mixing),
        gptneox_rope_attention_score(base, dim, mask, p),
        head, q, k, v, position_embedding)
end

struct CausalGPTNeoXRoPEMultiheadQKVAttenOp{F, D, P} <: AbstractAttenOp
    base::F
    dim::D
    head::Int
    p::P
end
CausalGPTNeoXRoPEMultiheadQKVAttenOp(dim::Int, head::Int) = CausalGPTNeoXRoPEMultiheadQKVAttenOp(1e4, dim, head, nothing)
CausalGPTNeoXRoPEMultiheadQKVAttenOp(base, dim::Int, head::Int) = CausalGPTNeoXRoPEMultiheadQKVAttenOp(base, dim, head, nothing)
NeuralAttentionlib.get_attention_func(::CausalGPTNeoXRoPEMultiheadQKVAttenOp) = gptneox_rope_multihead_qkv_attention
NeuralAttentionlib.get_attention_func_args(op::CausalGPTNeoXRoPEMultiheadQKVAttenOp, q, k, v, mask = nothing) =
    (op.base, op.dim, op.head, q, k, v, BatchedMask(CausalMask() & mask), op.p)

Layers.set_dropout(op::CausalGPTNeoXRoPEMultiheadQKVAttenOp, p) = CausalGPTNeoXRoPEMultiheadQKVAttenOp(op.base, op.dim, op.head, p)

const CausalGPTNeoXRoPEMultiheadQKVAttenOpWithScore{F, D, P} = NeuralAttentionlib.WithScore{CausalGPTNeoXRoPEMultiheadQKVAttenOp{F, D, P}}

Layers.argument_names(::CausalGPTNeoXRoPEMultiheadQKVAttenOp) = (:hidden_state, :attention_mask)
Layers.apply_on_namedtuple(op::CausalGPTNeoXRoPEMultiheadQKVAttenOp, nt::NamedTuple) = Layers.apply_attention_op(op, nt)
