using ..Layers
using ..Layers: CompositeEmbedding, SelfAttention, CausalMultiheadQKVAttenOp
using Functors
using Static

using NeuralAttentionlib
using NeuralAttentionlib: WithScore

@hgfdef GPT2 (
    Model => (embed, decoder),
    LMHeadModel,
    # DoubleHeadsModel,
)

basemodelkey(::Type{<:HGFPreTrained{:gpt2}}) = :transformer


function load_model(_type::Type{HGFGPT2Model}, cfg, state_dict, prefix)
    embed = load_model(_type, CompositeEmbedding, cfg, state_dict, prefix)
    decoder = load_model(_type, TransformerBlock, cfg, state_dict, prefix)
    return HGFGPT2Model(embed, decoder)
end

function load_model(_type::Type{HGFGPT2LMHeadModel}, cfg, state_dict, prefix)
    model = load_model(HGFGPT2Model, cfg, state_dict, joinname(prefix, "transformer"))
    if cfg[:tie_word_embeddings]
        embedding = model.embed.layer.token.embeddings
    else
        vocab_size, dims, factor = cfg[:vocab_size], cfg[:n_embd], Float32(cfg[:initializer_range])
        embedding = getweight(weight_init(vocab_size, dims, factor), Layers.Embed,
                              state_dict, joinname(prefix, "lm_head.weight"))
    end
    lmhead = Layers.EmbedDecoder(Layers.Embed(embedding))
    return HGFGPT2LMHeadModel(model, Layers.Branch{(:logit,), (:hidden_state,)}(lmhead))
end

function load_model(_type::Type{<:HGFGPT2PreTrainedModel}, ::Type{<:CompositeEmbedding}, cfg, state_dict, prefix)
    vocab_size, dims, max_pos = cfg[:vocab_size], cfg[:n_embd], cfg[:n_positions]
    factor = Float32(cfg[:initializer_range])
    p = cfg[:embd_pdrop]; p = iszero(p) ? nothing : p
    token_weight = getweight(weight_init(vocab_size, dims, factor), Layers.Embed,
                             state_dict, joinname(prefix, "wte.weight"))
    pos_weight = getweight(weight_init(max_pos, dims, factor), Layers.Embed, state_dict, joinname(prefix, "wpe.weight"))
    embed = CompositeEmbedding(
        token = Layers.Embed(token_weight),
        position = Layers.FixedLenPositionEmbed(pos_weight)
    )
    return Layers.DropoutLayer(embed, p)
end

function load_model(_type::Type{<:HGFGPT2PreTrainedModel}, ::Type{<:Layers.LayerNorm}, cfg, state_dict, prefix)
    dims = cfg[:n_embd]
    ln_ϵ = Float32(cfg[:layer_norm_epsilon])
    old_weight_name = joinname(prefix, "gamma")
    old_bias_name = joinname(prefix, "beta")
    weight_name = haskey(state_dict, old_weight_name) ? old_weight_name : joinname(prefix, "weight")
    bias_name = haskey(state_dict, old_bias_name) ? old_bias_name : joinname(prefix, "bias")
    ln_weight = getweight(one_init(dims), Array, state_dict, weight_name)
    ln_bias = getweight(zero_init(dims), Array, state_dict, bias_name)
    return Layers.LayerNorm(ln_weight, ln_bias, ln_ϵ)
end

function load_model(_type::Type{<:HGFGPT2PreTrainedModel}, ::Type{<:SelfAttention}, cfg, state_dict, prefix)
    head, dims = cfg[:n_head], cfg[:n_embd]
    @assert dims % head == 0 "The hidden size is not a multiple of the number of attention heads."
    p = cfg[:attn_pdrop]; p = iszero(p) ? nothing : p
    return_score = cfg[:output_attentions]
    factor = Float32(cfg[:initializer_range])
    # TODO: cfg[:scale_attn_weights]
    attn_weight = getweight(weight_init(dims, 3dims, factor), Layers.Embed, state_dict, joinname(prefix, "c_attn.weight"))
    attn_bias = getweight(zero_init(3dims), Array, state_dict, joinname(prefix, "c_attn.bias"))
    qkv_proj = Layers.NSplit(static(3), Layers.Dense(attn_weight, attn_bias))
    o_weight = getweight(weight_init(dims, dims, factor), Layers.Embed, state_dict, joinname(prefix, "c_proj.weight"))
    o_bias = getweight(zero_init(dims), Array, state_dict, joinname(prefix, "c_proj.bias"))
    o_proj = Layers.Dense(o_weight, o_bias)
    op = CausalMultiheadQKVAttenOp(head, p)
    return_score && (op = WithScore(op))
    return SelfAttention(op, qkv_proj, o_proj)
end

function load_model(
    _type::Type{<:HGFGPT2PreTrainedModel}, ::Type{<:Layers.Chain{<:Tuple{Layers.Dense, Layers.Dense}}},
    cfg, state_dict, prefix
)
    dims = cfg[:n_embd]
    ff_dims = get(cfg, :n_inner, nothing)
    ff_dims = isnothing(ff_dims) ? 4dims : ff_dims
    factor = Float32(cfg[:initializer_range])
    act = ACT2FN[Symbol(cfg[:activation_function])]
    wi_weight = getweight(weight_init(dims, ff_dims, factor), Layers.Embed,
                          state_dict, joinname(prefix, "c_fc.weight"))
    wi_bias = getweight(zero_init(ff_dims), Array, state_dict, joinname(prefix, "c_fc.bias"))
    wo_weight = getweight(weight_init(ff_dims, dims, factor), Layers.Embed,
                          state_dict, joinname(prefix, "c_proj.weight"))
    wo_bias = getweight(zero_init(dims), Array, state_dict, joinname(prefix, "c_proj.bias"))
    return Layers.Chain(Layers.Dense(act, wi_weight, wi_bias), Layers.Dense(wo_weight, wo_bias))
end

function load_model(_type::Type{<:HGFGPT2PreTrainedModel}, ::Type{<:TransformerBlock}, cfg, state_dict, prefix)
    n = cfg[:n_layer]
    p = cfg[:resid_pdrop]; p = iszero(p) ? nothing : p
    collect_output = cfg[:output_attentions] || cfg[:output_hidden_states]
    blocks = []
    for i = 1:n
        lprefix = joinname(prefix, :h, i-1)
        sa = load_model(_type, SelfAttention, cfg, state_dict, joinname(lprefix, "attn"))
        sa_ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(lprefix, "ln_1"))
        sa = Layers.PreNormResidual(Layers.DropoutLayer(sa, p), sa_ln)
        ff = load_model(_type, Layers.Chain{Tuple{Layers.Dense, Layers.Dense}}, cfg, state_dict, joinname(lprefix, "mlp"))
        ff_ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(lprefix, "ln_2"))
        ff = Layers.PreNormResidual(Layers.DropoutLayer(ff, p), ff_ln)
        block = TransformerBlock(sa, ff)
        push!(blocks, block)
    end
    collect_f = collect_output ? Layers.collect_outputs : nothing
    trf = Transformer(Tuple(blocks), collect_f)
    final_ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(prefix, "ln_f"))
    return Layers.Chain(trf, final_ln)
end

function get_state_dict(m::HGFGPT2Model, state_dict, prefix)
    get_state_dict(HGFGPT2Model, m.embed, state_dict, prefix)
    get_state_dict(HGFGPT2Model, m.decoder[1], state_dict, prefix)
    get_state_dict(HGFGPT2Model, m.decoder[2], state_dict, joinname(prefix, "ln_f"))
    return state_dict
end

function get_state_dict(m::HGFGPT2LMHeadModel, state_dict, prefix)
    get_state_dict(m.model, state_dict, joinname(prefix, "transformer"))
    get_state_dict(HGFGPT2Model, m.cls.layer.embed, state_dict, joinname(prefix, "lm_head"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFGPT2PreTrainedModel}, m::CompositeEmbedding, state_dict, prefix)
    get_state_dict(p, m.token, state_dict, joinname(prefix, "wte"))
    get_state_dict(p, m.position.embed, state_dict, joinname(prefix, "wpe"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFGPT2PreTrainedModel}, m::Layers.Dense, state_dict, prefix)
    state_dict[joinname(prefix, "weight")] = m.W'
    state_dict[joinname(prefix, "bias")] = m.b
    return state_dict
end

function get_state_dict(p::Type{<:HGFGPT2PreTrainedModel}, m::SelfAttention, state_dict, prefix)
    get_state_dict(p, m.qkv_proj.layer, state_dict, joinname(prefix, "c_attn"))
    get_state_dict(p, m.o_proj, state_dict, joinname(prefix, "c_proj"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFGPT2PreTrainedModel}, m::Layers.Chain{<:Tuple{Layers.Dense, Layers.Dense}},
                        state_dict, prefix)
    get_state_dict(p, m[1], state_dict, joinname(prefix, "c_fc"))
    get_state_dict(p, m[2], state_dict, joinname(prefix, "c_proj"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFGPT2PreTrainedModel}, m::TransformerBlock, state_dict, prefix)
    get_state_dict(p, m.attention.layer, state_dict, joinname(prefix, "attn"))
    get_state_dict(p, m.attention.norm, state_dict, joinname(prefix, "ln_1"))
    get_state_dict(p, m.feedforward.layer, state_dict, joinname(prefix, "mlp"))
    get_state_dict(p, m.feedforward.norm, state_dict, joinname(prefix, "ln_2"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFGPT2PreTrainedModel}, m::Transformer, state_dict, prefix)
    for (i, t) in enumerate(m.blocks)
        get_state_dict(p, t, state_dict, joinname(prefix, :h, i-1))
    end
    return state_dict
end
