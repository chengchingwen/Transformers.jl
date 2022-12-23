using ..Layers
using ..Layers: CompositeEmbedding, SelfAttention
using ChainRulesCore
using Functors
using Static

using NeuralAttentionlib
using NeuralAttentionlib: $, AbstractAttenOp, WithScore, with_rotary_position_embedding,
    scaled_dot_product_score, masked_score, normalized_score, dropout_score, weighted_sum_mixing,
    generic_multihead_qkv_attention, CausalMask, BatchedMask

rope_attention(dim, mask, p) =
    dropout_score(p) $
    normalized_score(softmax) $
    masked_score(NeuralAttentionlib.GenericMaskOp(), mask) $
    scaled_dot_product_score $
    with_rotary_position_embedding(dim)

ChainRulesCore.@non_differentiable rope_attention(arg...)

function rope_multihead_qkv_attention(dim, head, q, k, v, mask = nothing, p = nothing)
    return generic_multihead_qkv_attention(
        weighted_sum_mixing, rope_attention(dim, mask, p),
        head, q, k, v)
end

function rope_multihead_qkv_attention(
    ::typeof(NeuralAttentionlib.score_returning),
    dim, head, q, k, v, mask = nothing, p = nothing
)
    return generic_multihead_qkv_attention(
        NeuralAttentionlib.score_returning(weighted_sum_mixing),
        rope_attention(dim, mask, p),
        head, q, k, v, position_embedding)
end

struct CausalRoPEMultiheadQKVAttenOp{D, F} <: AbstractAttenOp
    dim::D
    head::Int
    p::F
end
CausalRoPEMultiheadQKVAttenOp(head::Int) = CausalRoPEMultiheadQKVAttenOp(nothing, head, nothing)
CausalRoPEMultiheadQKVAttenOp(dim::Int, head::Int) = CausalRoPEMultiheadQKVAttenOp(dim, head, nothing)
NeuralAttentionlib.get_attention_func(::CausalRoPEMultiheadQKVAttenOp) = rope_multihead_qkv_attention
NeuralAttentionlib.get_attention_func_args(op::CausalRoPEMultiheadQKVAttenOp, q, k, v, mask = nothing) = (op.dim, op.head, q, k, v, BatchedMask(mask & CausalMask()), op.p)

Layers.no_dropout(op::CausalRoPEMultiheadQKVAttenOp) = CausalRoPEMultiheadQKVAttenOp(op.dim, op.head, nothing)

const CausalRoPEMultiheadQKVAttenOpWithScore{D, F} = WithScore{CausalRoPEMultiheadQKVAttenOp{D, F}}

Layers.argument_names(
    ::Union{CausalRoPEMultiheadQKVAttenOpWithScore, CausalRoPEMultiheadQKVAttenOp}
) = (:hidden_state, :attention_mask)

function Layers.apply_on_namedtuple(
    op::Union{CausalRoPEMultiheadQKVAttenOpWithScore, CausalRoPEMultiheadQKVAttenOp},
    nt::NamedTuple
)
    return Layers.apply_attention_op(op, nt)
end

struct ParallelPreNormTransformerBlock{A, F, N} <: Layers.AbstractTransformerBlock
    attention::A
    feedforward::F
    norm::N
end
@functor ParallelPreNormTransformerBlock

function (b::ParallelPreNormTransformerBlock)(nt::NamedTuple)
    nt2 = Layers.apply_on_namedtuple(b.norm, nt)
    a = Layers.apply_on_namedtuple(b.attention, nt2)
    f = Layers.apply_on_namedtuple(b.feedforward, nt2)
    hidden_state = a.hidden_state + f.hidden_state + nt.hidden_state
    return Layers.return_hidden_state(a, hidden_state)
end

abstract type HGFGPTJPreTrainedModel <: HGFPreTrainedModel end

struct HGFGPTJModel{E, DEC} <: HGFGPTJPreTrainedModel
    embed::E
    decoder::DEC
end
@functor HGFGPTJModel

(model::HGFGPTJModel)(nt::NamedTuple) = model.decoder(model.embed(nt))

for T in :[
    HGFGPTJForCausalLM
].args
    @eval begin
        struct $T{M, C} <: HGFGPTJPreTrainedModel
            model::M
            cls::C
        end
        @functor $T
        (model::$T)(nt::NamedTuple) = model.cls(model.model(nt))
    end
end

for T in :[
    ParallelPreNormTransformerBlock, HGFGPTJModel, HGFGPTJForCausalLM
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

basemodelkey(::Type{<:HGFGPTJPreTrainedModel}) = :transformer
isbasemodel(::Type{<:HGFGPTJModel}) = true
isbasemodel(::Type{<:HGFGPTJPreTrainedModel}) = false

get_model_type(::Val{:gptj}) = (
    model = HGFGPTJModel,
    forcausallm = HGFGPTJForCausalLM,
)

function load_model(_type::Type{HGFGPTJModel}, cfg, state_dict, prefix)
    embed = load_model(_type, CompositeEmbedding, cfg, state_dict, prefix)
    decoder = load_model(_type, TransformerBlock, cfg, state_dict, prefix)
    return HGFGPTJModel(embed, decoder)
end

function load_model(_type::Type{HGFGPTJForCausalLM}, cfg, state_dict, prefix)
    model = load_model(HGFGPTJModel, cfg, state_dict, joinname(prefix, "transformer"))
    if cfg[:tie_word_embeddings]
        embedding = model.embed.layer.token.embeddings
    else
        vocab_size, dims, factor = cfg[:vocab_size], cfg[:n_embd], Float32(cfg[:initializer_range])
        embedding = getweight(weight_init(vocab_size, dims, factor), Layers.Embed,
                              state_dict, joinname(prefix, "lm_head.weight"))
    end
    lmhead = Layers.EmbedDecoder(Layers.Embed(embedding))
    return HGFGPTJForCausalLM(model, Layers.Branch{(:logit,), (:hidden_state,)}(lmhead))
end


function load_model(_type::Type{<:HGFGPTJPreTrainedModel}, ::Type{<:CompositeEmbedding}, cfg, state_dict, prefix)
    vocab_size, dims = cfg[:vocab_size], cfg[:n_embd]
    factor = Float32(cfg[:initializer_range])
    p = cfg[:embd_pdrop]; p = iszero(p) ? nothing : p
    token_weight = getweight(weight_init(vocab_size, dims, factor), Layers.Embed, state_dict, joinname(prefix, "wte.weight"))
    embed = CompositeEmbedding(token = Layers.Embed(token_weight))
    return Layers.DropoutLayer(embed, p)
end

function load_model(_type::Type{<:HGFGPTJPreTrainedModel}, ::Type{<:Layers.LayerNorm}, cfg, state_dict, prefix)
    dims = cfg[:n_embd]
    ln_ϵ = Float32(cfg[:layer_norm_epsilon])
    old_weight_name = joinname(prefix, "gamma")
    old_bias_name = joinname(prefix, "beta")
    weight_name = haskey(state_dict, old_weight_name) ? old_weight_name : joinname(prefix, "weight")
    bias_name = haskey(state_dict, old_bias_name) ? old_bias_name : joinname(prefix, "bias")
    ln_weight = getweight(() -> ones(Float32, dims), Array, state_dict, weight_name)
    ln_bias = getweight(() -> zeros(Float32, dims), Array, state_dict, bias_name)
    return Layers.LayerNorm(ln_weight, ln_bias, ln_ϵ)
end

function load_model(_type::Type{<:HGFGPTJPreTrainedModel}, ::Type{<:SelfAttention}, cfg, state_dict, prefix)
    head, dims = cfg[:n_head], cfg[:n_embd]
    @assert dims % head == 0 "The hidden size is not a multiple of the number of attention heads."
    p = cfg[:attn_pdrop]; p = iszero(p) ? nothing : p
    return_score = cfg[:output_attentions]
    factor = Float32(cfg[:initializer_range])
    rotary_dim = get(cfg, :rotary_dim, nothing)
    q_weight = getweight(weight_init(dims, dims, factor), Array, state_dict, joinname(prefix, "q_proj.weight"))
    k_weight = getweight(weight_init(dims, dims, factor), Array, state_dict, joinname(prefix, "k_proj.weight"))
    v_weight = getweight(weight_init(dims, dims, factor), Array, state_dict, joinname(prefix, "v_proj.weight"))
    qkv_proj = Layers.Fork(Layers.Dense(q_weight), Layers.Dense(k_weight), Layers.Dense(v_weight))
    o_weight = getweight(weight_init(dims, dims, factor), Array, state_dict, joinname(prefix, "out_proj.weight"))
    o_proj = Layers.Dense(o_weight)
    op = CausalRoPEMultiheadQKVAttenOp(rotary_dim, head, p)
    return_score && (op = WithScore(op))
    return SelfAttention(op, qkv_proj, o_proj)
end

function load_model(
    _type::Type{<:HGFGPTJPreTrainedModel}, ::Type{<:Layers.Chain{<:Tuple{Layers.Dense, Layers.Dense}}},
    cfg, state_dict, prefix
)
    dims = cfg[:n_embd]
    ff_dims = get(cfg, :n_inner, nothing)
    ff_dims = isnothing(ff_dims) ? 4dims : ff_dims
    factor = Float32(cfg[:initializer_range])
    act = ACT2FN[Symbol(cfg[:activation_function])]
    wi_weight = getweight(weight_init(dims, ff_dims, factor), Array, state_dict, joinname(prefix, "fc_in.weight"))
    wi_bias = getweight(zero_init(ff_dims), Array, state_dict, joinname(prefix, "fc_in.bias"))
    wo_weight = getweight(weight_init(ff_dims, dims, factor), Array, state_dict, joinname(prefix, "fc_out.weight"))
    wo_bias = getweight(zero_init(dims), Array, state_dict, joinname(prefix, "fc_out.bias"))
    return Layers.Chain(Layers.Dense(act, wi_weight, wi_bias), Layers.Dense(wo_weight, wo_bias))
end

function load_model(_type::Type{<:HGFGPTJPreTrainedModel}, ::Type{<:TransformerBlock}, cfg, state_dict, prefix)
    n = cfg[:n_layer]
    p = cfg[:resid_pdrop]; p = iszero(p) ? nothing : p
    collect_output = cfg[:output_attentions] || cfg[:output_hidden_states]
    blocks = []
    for i = 1:n
        lprefix = joinname(prefix, :h, i-1)
        ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(lprefix, "ln_1"))
        sa = load_model(_type, SelfAttention, cfg, state_dict, joinname(lprefix, "attn"))
        sa = Layers.DropoutLayer(sa, p)
        ff = load_model(_type, Layers.Chain{Tuple{Layers.Dense, Layers.Dense}}, cfg, state_dict, joinname(lprefix, "mlp"))
        ff = Layers.DropoutLayer(ff, p)
        block = ParallelPreNormTransformerBlock(sa, ff, ln)
            TransformerBlock(sa, ff)
        push!(blocks, block)
    end
    collect_f = collect_output ? Layers.collect_outputs : nothing
    trf = Transformer(Tuple(blocks), collect_f)
    final_ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(prefix, "ln_f"))
    return Layers.Chain(trf, final_ln)
end
