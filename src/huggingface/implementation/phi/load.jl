using ..Layers
using ..Layers: CompositeEmbedding, SelfAttention
using ChainRulesCore
using Functors
using Static

using NeuralAttentionlib
using NeuralAttentionlib: WithScore

@hgfdef Phi (
    Model => (embed, decoder),
    ForCausalLM,
)

basemodelkey(::Type{<:HGFPhiPreTrainedModel}) = :model

function load_model(_type::Type{HGFPhiModel}, cfg, state_dict, prefix)
    embed = load_model(_type, CompositeEmbedding, cfg, state_dict, prefix)
    decoder = load_model(_type, TransformerBlock, cfg, state_dict, prefix)
    return HGFPhiModel(embed, decoder)
end

function load_model(_type::Type{HGFPhiForCausalLM}, cfg, state_dict, prefix)
    model = load_model(HGFPhiModel, cfg, state_dict, joinname(prefix, "model"))
    if cfg[:tie_word_embeddings]
        embedding = model.embed.token.embeddings
    else
        vocab_size, dims, factor = cfg[:vocab_size], cfg[:hidden_size], Float32(cfg[:initializer_range])
        embedding = getweight(weight_init(vocab_size, dims, factor), Layers.Embed,
                              state_dict, joinname(prefix, "lm_head.weight"))
    end
    bias = getweight(zero_init(vocab_size), Array, state_dict, joinname(prefix, "lm_head.bias"))
    lmhead = Layers.EmbedDecoder(Layers.Embed(embedding), bias)
    return HGFPhiForCausalLM(model, Layers.Branch{(:logit,), (:hidden_state,)}(lmhead))
end

function load_model(_type::Type{<:HGFPhiPreTrainedModel}, ::Type{<:CompositeEmbedding}, cfg, state_dict, prefix)
    vocab_size, dims = cfg[:vocab_size], cfg[:hidden_size]
    factor = Float32(cfg[:initializer_range])
    token_weight = getweight(weight_init(vocab_size, dims, factor), Layers.Embed,
                             state_dict, joinname(prefix, "embed_tokens.weight"))
    p = cfg[:embd_pdrop]; p = iszero(p) ? nothing : p
    embed = CompositeEmbedding(token = Layers.Embed(token_weight))
    return Layers.DropoutLayer(embed, p)
end

function load_model(
    _type::Type{<:HGFPhiPreTrainedModel}, ::Type{<:Layers.Chain{<:Tuple{Layers.Dense, Layers.Dense}}},
    cfg, state_dict, prefix
)
    dims = cfg[:hidden_size]
    ff_dims = cfg[:intermediate_size]
    factor = Float32(cfg[:initializer_range])
    act = ACT2FN[Symbol(cfg[:hidden_act])]
    wi_weight = getweight(weight_init(dims, ff_dims, factor), Array,
                          state_dict, joinname(prefix, "fc1.weight"))
    wi_bias = getweight(zero_init(ff_dims), Array, state_dict, joinname(prefix, "fc1.bias"))
    wo_weight = getweight(weight_init(ff_dims, dims, factor), Array,
                          state_dict, joinname(prefix, "fc2.weight"))
    wo_bias = getweight(zero_init(dims), Array, state_dict, joinname(prefix, "fc2.bias"))
    return Layers.Chain(Layers.Dense(act, wi_weight, wi_bias), Layers.Dense(wo_weight, wo_bias))
end

function load_model(_type::Type{<:HGFPhiPreTrainedModel}, ::Type{<:SelfAttention}, cfg, state_dict, prefix)
    head, dims = cfg[:num_attention_heads], cfg[:hidden_size]
    @assert dims % head == 0 "The hidden size is not a multiple of the number of attention heads."
    p = cfg[:attention_dropout]; p = iszero(p) ? nothing : p
    head_dims = div(dims, head)
    kv_head = something(cfg[:num_key_value_heads], head)
    grouped_attn = kv_head != head
    @assert head % kv_head == 0 "The number of query is not dividable by the number of key/value groups"
    return_score = cfg[:output_attentions]
    factor = Float32(cfg[:initializer_range])
    rotary_dim = floor(Int, cfg[:partial_rotary_factor] * head_dims)
    rotary_pe_base = Float64(cfg[:rope_theta])
    @assert isnothing(cfg[:rope_scaling]) "Scaling Rotary Embedding is not support yet"
    kv_dims = kv_head * head_dims
    q_weight = getweight(weight_init(dims, dims, factor), Array, state_dict, joinname(prefix, "q_proj.weight"))
    q_bias = getweight(zero_init(dims), Array, state_dict, joinname(prefix, "q_proj.bias"))
    k_weight = getweight(weight_init(dims, kv_dims, factor), Array, state_dict, joinname(prefix, "k_proj.weight"))
    k_bias = getweight(zero_init(kv_dims), Array, state_dict, joinname(prefix, "k_proj.bias"))
    v_weight = getweight(weight_init(dims, kv_dims, factor), Array, state_dict, joinname(prefix, "v_proj.weight"))
    v_bias = getweight(zero_init(kv_dims), Array, state_dict, joinname(prefix, "v_proj.bias"))
    o_weight = getweight(weight_init(dims, dims, factor), Array, state_dict, joinname(prefix, "dense.weight"))
    o_bias = getweight(zero_init(dims), Array, state_dict, joinname(prefix, "dense.bias"))
    query = Layers.Dense(q_weight, q_bias)
    key = Layers.Dense(k_weight, k_bias)
    value = Layers.Dense(v_weight, v_bias)
    if cfg[:qk_layernorm]
        ln_ϵ = Float32(cfg[:layer_norm_eps])
        q_layernorm = _load_layernorm(state_dict, joinname(lprefix, "q_layernorm"), head_dims, ln_ϵ)
        k_layernorm = _load_layernorm(state_dict, joinname(lprefix, "k_layernorm"), head_dims, ln_ϵ)
        query = Layers.Chain(query, q_layernorm)
        key = Layers.Chain(key, k_layernorm)
    end
    qkv_proj = Layers.Fork(query, key, value)
    o_proj = Layers.Dense(o_weight, o_bias)
    if grouped_attn
        op = CausalLlamaRoPEGroupedQueryAttenOp(rotary_pe_base, rotary_dim, head, kv_head, p)
    else
        op = CausalGPTNeoXRoPEMultiheadQKVAttenOp(rotary_pe_base, rotary_dim, head, p)
    end
    return_score && (op = WithScore(op))
    return SelfAttention(op, qkv_proj, o_proj)
end

function load_model(::Type{<:HGFPhiPreTrainedModel}, ::Type{<:Layers.LayerNorm}, cfg, state_dict, prefix)
    dims = cfg[:hidden_size]
    ln_ϵ = Float32(cfg[:layer_norm_eps])
    ln_weight = getweight(one_init(dims), Array, state_dict, joinname(prefix, "weight"))
    ln_bias = getweight(zero_init(dims), Array, state_dict, joinname(prefix, "bias"))
    return Layers.LayerNorm(ln_weight, ln_bias, ln_ϵ)
end

function load_model(_type::Type{<:HGFPhiPreTrainedModel}, ::Type{<:TransformerBlock}, cfg, state_dict, prefix)
    n = cfg[:num_hidden_layers]
    p = cfg[:resid_pdrop]; p = iszero(p) ? nothing : p
    collect_output = cfg[:output_attentions] || cfg[:output_hidden_states]
    blocks = []
    for i = 1:n
        lprefix = joinname(prefix, :layers, i-1)
        ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(lprefix, "input_layernorm"))
        sa = load_model(_type, SelfAttention, cfg, state_dict, joinname(lprefix, "self_attn"))
        sa = Layers.DropoutLayer(sa, p)
        ff = load_model(_type, Layers.Chain{Tuple{Layers.Dense, Layers.Dense}}, cfg, state_dict, joinname(lprefix, "mlp"))
        ff = Layers.DropoutLayer(ff, p)
        block = ParallelPreNormTransformerBlock(sa, ff, ln)
        push!(blocks, block)
    end
    collect_f = collect_output ? Layers.collect_outputs : nothing
    trf = Transformer(Tuple(blocks), collect_f)
    final_ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(prefix, "final_layernorm"))
    return Layers.Chain(trf, final_ln)
end


function get_state_dict(m::HGFPhiModel, state_dict, prefix)
    get_state_dict(HGFPhiModel, m.embed, state_dict, prefix)
    get_state_dict(HGFPhiModel, m.decoder[1], state_dict, prefix)
    get_state_dict(HGFPhiModel, m.decoder[2], state_dict, joinname(prefix, "final_layernorm"))
    return state_dict
end

function get_state_dict(m::HGFPhiForCausalLM, state_dict, prefix)
    get_state_dict(m.model, state_dict, joinname(prefix, "model"))
    get_state_dict(HGFPhiModel, m.cls.layer, state_dict, joinname(prefix, "lm_head"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFPhiPreTrainedModel}, m::CompositeEmbedding, state_dict, prefix)
    get_state_dict(p, m.token, state_dict, joinname(prefix, "embed_tokens"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFPhiPreTrainedModel}, m::SelfAttention, state_dict, prefix)
    q, k, v = m.qkv_proj.layers
    if q isa Layers.Chain
        get_state_dict(p, q[1], state_dict, joinname(prefix, "q_proj"))
        get_state_dict(p, k[1], state_dict, joinname(prefix, "k_proj"))
        get_state_dict(p, q[2], state_dict, joinname(prefix, "q_layernorm"))
        get_state_dict(p, k[2], state_dict, joinname(prefix, "k_layernorm"))
    else
        get_state_dict(p, q, state_dict, joinname(prefix, "q_proj"))
        get_state_dict(p, k, state_dict, joinname(prefix, "k_proj"))
    end
    get_state_dict(p, v, state_dict, joinname(prefix, "v_proj"))
    get_state_dict(p, m.o_proj, state_dict, joinname(prefix, "dense"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFPhiPreTrainedModel}, m::Layers.Chain{<:Tuple{Layers.Dense, Layers.Dense}},
                        state_dict, prefix)
    get_state_dict(p, m[1], state_dict, joinname(prefix, "fc1"))
    get_state_dict(p, m[2], state_dict, joinname(prefix, "fc2"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFPhiPreTrainedModel}, m::ParallelPreNormTransformerBlock, state_dict, prefix)
    get_state_dict(p, m.norm, state_dict, joinname(prefix, "input_layernorm"))
    get_state_dict(p, m.attention, state_dict, joinname(prefix, "self_attn"))
    get_state_dict(p, m.feedforward, state_dict, joinname(prefix, "mlp"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFPhiPreTrainedModel}, m::Transformer, state_dict, prefix)
    for (i, t) in enumerate(m.blocks)
        get_state_dict(p, t, state_dict, joinname(prefix, :layers, i-1))
    end
    return state_dict
end
