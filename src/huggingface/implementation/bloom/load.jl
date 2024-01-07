using ..Layers
using ..Layers: CompositeEmbedding, SelfAttention, CausalALiBiMultiheadQKVAttenOp
using ChainRulesCore
using Functors
using Static

using NeuralAttentionlib
using NeuralAttentionlib: WithScore

@hgfdef Bloom (
    Model => (embed, decoder),
    ForCausalLM,
    # ForSequenceClassification,
    # ForTokenClassification,
    # ForQuestionAnswering,
)

basemodelkey(::Type{<:HGFPreTrained{:bloom}}) = :transformer

function load_model(_type::Type{HGFBloomModel}, cfg, state_dict, prefix)
    embed = load_model(_type, CompositeEmbedding, cfg, state_dict, prefix)
    decoder = load_model(_type, TransformerBlock, cfg, state_dict, prefix)
    return HGFBloomModel(embed, decoder)
end

function load_model(_type::Type{HGFBloomForCausalLM}, cfg, state_dict, prefix)
    model = load_model(HGFBloomModel, cfg, state_dict, joinname(prefix, "transformer"))
    if cfg[:tie_word_embeddings]
        embed = model.embed[1].token
    else
        vocab_size, dims, factor = cfg[:vocab_size], cfg[:hidden_size], Float32(cfg[:initializer_range])
        embed = _load_embed(state_dict, joinname(prefix, "lm_head"), vocab_size, dims, factor)
    end
    lmhead = Layers.EmbedDecoder(embed)
    return HGFBloomForCausalLM(model, Layers.Branch{(:logit,), (:hidden_state,)}(lmhead))
end

function load_model(_type::Type{<:HGFBloomPreTrainedModel}, ::Type{<:CompositeEmbedding}, cfg, state_dict, prefix)
    vocab_size, dims = cfg[:vocab_size], cfg[:hidden_size]
    factor = Float32(cfg[:initializer_range])
    token_embed = _load_embed(state_dict, joinname(prefix, "word_embeddings"), vocab_size, dims, factor)
    embed = CompositeEmbedding(token = token_embed)
    ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(prefix, "word_embeddings_layernorm"))
    return Layers.Chain(embed, ln)
end

function load_model(_type::Type{<:HGFBloomPreTrainedModel}, ::Type{<:Layers.LayerNorm}, cfg, state_dict, prefix)
    dims = cfg[:hidden_size]
    ln_ϵ = Float32(cfg[:layer_norm_epsilon])
    return _load_layernorm(state_dict, prefix, dims, ln_ϵ)
end

function load_model(_type::Type{<:HGFBloomPreTrainedModel}, ::Type{<:SelfAttention}, cfg, state_dict, prefix)
    head, dims = cfg[:n_head], cfg[:hidden_size]
    @assert dims % head == 0 "The hidden size is not a multiple of the number of attention heads."
    p = cfg[:attention_dropout]; p = iszero(p) ? nothing : p
    return_score = cfg[:output_attentions]
    factor = Float32(cfg[:initializer_range])
    qkv_proj = GPTNeoXSplit(head, _load_dense(state_dict, joinname(prefix, "query_key_value"), dims, 3dims, factor, true))
    o_proj = _load_dense(state_dict, joinname(prefix, "dense"), dims, dims, factor, true)
    op = CausalALiBiMultiheadQKVAttenOp(head, p)
    return_score && (op = WithScore(op))
    return SelfAttention(op, qkv_proj, o_proj)
end

function load_model(
    _type::Type{<:HGFBloomPreTrainedModel}, ::Type{<:Layers.Chain{<:Tuple{Layers.Dense, Layers.Dense}}},
    cfg, state_dict, prefix
)
    dims = cfg[:hidden_size]
    factor = Float32(cfg[:initializer_range])
    return Layers.Chain(
        _load_dense(state_dict, joinname(prefix, "dense_h_to_4h"), dims, 4dims, factor, true, gelu),
        _load_dense(state_dict, joinname(prefix, "dense_4h_to_h"), 4dims, dims, factor, true)
    )
end

function load_model(_type::Type{<:HGFBloomPreTrainedModel}, ::Type{<:TransformerBlock}, cfg, state_dict, prefix)
    n = cfg[:num_hidden_layers]
    p = cfg[:hidden_dropout]; p = iszero(p) ? nothing : p
    resi_post_ln = cfg[:apply_residual_connection_post_layernorm]
    collect_output = cfg[:output_attentions] || cfg[:output_hidden_states]
    blocks = []
    for i = 1:n
        lprefix = joinname(prefix, :h, i-1)
        sa_ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(lprefix, "input_layernorm"))
        sa = load_model(_type, SelfAttention, cfg, state_dict, joinname(lprefix, "self_attention"))
        sa = Layers.DropoutLayer(sa, p)
        if resi_post_ln
            sa = Layers.Chain(sa_ln, Layers.Residual(sa))
        else
            sa = Layers.PreNormResidual(sa, sa_ln)
        end
        ff_ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(lprefix, "post_attention_layernorm"))
        ff = load_model(_type, Layers.Chain{Tuple{Layers.Dense, Layers.Dense}}, cfg, state_dict, joinname(lprefix, "mlp"))
        ff = Layers.DropoutLayer(ff, p)
        if resi_post_ln
            ff = Layers.Chain(ff_ln, Layers.Residual(ff))
        else
            ff = Layers.PreNormResidual(ff, ff_ln)
        end
        block = TransformerBlock(sa, ff)
        push!(blocks, block)
    end
    collect_f = collect_output ? Layers.collect_outputs : nothing
    trf = Transformer(Tuple(blocks), collect_f)
    final_ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(prefix, "ln_f"))
    return Layers.Chain(trf, final_ln)
end

function get_state_dict(m::HGFBloomModel, state_dict, prefix)
    get_state_dict(HGFBloomModel, m.embed[1], state_dict, prefix)
    get_state_dict(HGFBloomModel, m.embed[2], state_dict, joinname(prefix, "word_embeddings_layernorm"))
    get_state_dict(HGFBloomModel, m.decoder[1], state_dict, prefix)
    get_state_dict(HGFBloomModel, m.decoder[2], state_dict, joinname(prefix, "ln_f"))
    return state_dict
end

function get_state_dict(m::HGFBloomForCausalLM, state_dict, prefix)
    get_state_dict(m.model, state_dict, joinname(prefix, "transformer"))
    get_state_dict(HGFBloomModel, m.cls.layer.embed, state_dict, joinname(prefix, "lm_head"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFBloomPreTrainedModel}, m::CompositeEmbedding, state_dict, prefix)
    get_state_dict(p, m.token, state_dict, joinname(prefix, "word_embeddings"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFBloomPreTrainedModel}, m::SelfAttention, state_dict, prefix)
    get_state_dict(p, m.qkv_proj.layer, state_dict, joinname(prefix, "query_key_value"))
    get_state_dict(p, m.o_proj, state_dict, joinname(prefix, "dense"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFBloomPreTrainedModel}, m::Layers.Chain{<:Tuple{Layers.Dense, Layers.Dense}},
                        state_dict, prefix)
    get_state_dict(p, m[1], state_dict, joinname(prefix, "dense_h_to_4h"))
    get_state_dict(p, m[2], state_dict, joinname(prefix, "dense_4h_to_h"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFBloomPreTrainedModel}, m::TransformerBlock, state_dict, prefix)
    if m.attention isa Layers.Chain
        get_state_dict(p, m.attention[1], state_dict, joinname(prefix, "input_layernorm"))
        get_state_dict(p, m.attention[2], state_dict, joinname(prefix, "self_attention"))
        get_state_dict(p, m.feedforward[1], state_dict, joinname(prefix, "post_attention_layernorm"))
        get_state_dict(p, m.feedforward[2], state_dict, joinname(prefix, "mlp"))
    else
        get_state_dict(p, m.attention.norm, state_dict, joinname(prefix, "input_layernorm"))
        get_state_dict(p, m.attention.layer, state_dict, joinname(prefix, "self_attention"))
        get_state_dict(p, m.feedforward.norm, state_dict, joinname(prefix, "post_attention_layernorm"))
        get_state_dict(p, m.feedforward.layer, state_dict, joinname(prefix, "mlp"))
    end
    return state_dict
end

function get_state_dict(p::Type{<:HGFBloomPreTrainedModel}, m::Transformer, state_dict, prefix)
    for (i, t) in enumerate(m.blocks)
        get_state_dict(p, t, state_dict, joinname(prefix, :h, i-1))
    end
    return state_dict
end
