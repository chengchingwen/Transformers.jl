using ..Layers
using ..Layers: CompositeEmbedding, SelfAttention, CausalRoPEMultiheadQKVAttenOp
using ChainRulesCore
using Functors
using Static

using NeuralAttentionlib
using NeuralAttentionlib: WithScore

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

@fluxshow ParallelPreNormTransformerBlock

@hgfdef GPTJ (
    Model => (embed, decoder),
    ForCausalLM,
)

basemodelkey(::Type{<:HGFPreTrained{:gptj}}) = :transformer

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
    ln_weight = getweight(one_init(dims), Array, state_dict, weight_name)
    ln_bias = getweight(zero_init(dims), Array, state_dict, bias_name)
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
        push!(blocks, block)
    end
    collect_f = collect_output ? Layers.collect_outputs : nothing
    trf = Transformer(Tuple(blocks), collect_f)
    final_ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(prefix, "ln_f"))
    return Layers.Chain(trf, final_ln)
end

function get_state_dict(m::HGFGPTJModel, state_dict, prefix)
    get_state_dict(HGFGPTJModel, m.embed, state_dict, prefix)
    get_state_dict(HGFGPTJModel, m.decoder[1], state_dict, prefix)
    get_state_dict(HGFGPTJModel, m.decoder[2], state_dict, joinname(prefix, "ln_f"))
    return state_dict
end

function get_state_dict(m::HGFGPTJForCausalLM, state_dict, prefix)
    get_state_dict(m.model, state_dict, joinname(prefix, "transformer"))
    get_state_dict(HGFGPTJModel, m.cls.layer.embed, state_dict, joinname(prefix, "lm_head"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFGPTJPreTrainedModel}, m::CompositeEmbedding, state_dict, prefix)
    get_state_dict(p, m.token, state_dict, joinname(prefix, "wte"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFGPTJPreTrainedModel}, m::SelfAttention, state_dict, prefix)
    get_state_dict(p, m.qkv_proj.layers[1], state_dict, joinname(prefix, "q_proj"))
    get_state_dict(p, m.qkv_proj.layers[2], state_dict, joinname(prefix, "k_proj"))
    get_state_dict(p, m.qkv_proj.layers[3], state_dict, joinname(prefix, "v_proj"))
    get_state_dict(p, m.o_proj, state_dict, joinname(prefix, "out_proj"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFGPTJPreTrainedModel}, m::Layers.Chain{<:Tuple{Layers.Dense, Layers.Dense}},
                        state_dict, prefix)
    get_state_dict(p, m[1], state_dict, joinname(prefix, "fc_in"))
    get_state_dict(p, m[2], state_dict, joinname(prefix, "fc_out"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFGPTJPreTrainedModel}, m::ParallelPreNormTransformerBlock, state_dict, prefix)
    get_state_dict(p, m.norm, state_dict, joinname(prefix, "ln_1"))
    get_state_dict(p, m.attention, state_dict, joinname(prefix, "attn"))
    get_state_dict(p, m.feedforward, state_dict, joinname(prefix, "mlp"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFGPTJPreTrainedModel}, m::Transformer, state_dict, prefix)
    for (i, t) in enumerate(m.blocks)
        get_state_dict(p, t, state_dict, joinname(prefix, :h, i-1))
    end
    return state_dict
end
