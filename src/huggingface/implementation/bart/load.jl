using ..Layers
using ..Layers: CompositeEmbedding, SelfAttention, CrossAttention, MultiheadQKVAttenOp, CausalMultiheadQKVAttenOp
using Functors
using NeuralAttentionlib
using NeuralAttentionlib: WithScore

bart_pe_shift(x) = bart_pe_shift(size(x, 2))
bart_pe_shift(len::Integer) = bart_pe_shift(Base.OneTo(len))
bart_pe_shift(x::AbstractArray{<:Integer}) = x .+ 2

@hgfdef Bart (
    Model => (embed, seq2seq),
    # ForConditionalGeneration,
    # ForSequenceClassification,
    # ForQuestionAnswering,
    # ForCausalLM,
)

basemodelkey(::Type{<:HGFBartPreTrainedModel}) = :model

function load_model(_type::Type{HGFBartModel}, cfg, state_dict, prefix)
    embed = load_model(_type, CompositeEmbedding, cfg, state_dict, prefix)
    encoder = load_model(_type, TransformerBlock, cfg, state_dict, joinname(prefix, "encoder"))
    decoder = load_model(_type, TransformerDecoderBlock, cfg, state_dict, joinname(prefix, "decoder"))
    seq2seq = Seq2Seq(encoder, decoder)
    return HGFBartModel(embed, seq2seq)
end

function load_model(::Type{<:HGFBartPreTrainedModel}, ::Type{Layers.Embed}, cfg, state_dict, prefix)
    vocab_size, dims, pad_id = cfg[:vocab_size], cfg[:d_model], cfg[:pad_token_id]
    factor = Float32(cfg[:init_std])
    scale = cfg.scale_embedding ? Float32(sqrt(dims)) : nothing
    token_weight = getweight(Layers.Embed, state_dict, joinname(prefix, "weight")) do
        weight = weight_init(vocab_size, dims, factor)()
        weight[:, pad_id+1] .= 0
        return weight
    end
    return Layers.Embed(scale, token_weight)
end
function load_model(::Type{<:HGFBartPreTrainedModel}, ::Type{Layers.FixedLenPositionEmbed}, cfg, state_dict, prefix)
    dims = cfg[:d_model]
    max_pos = cfg[:max_position_embeddings]
    factor = Float32(cfg[:init_std])
    pos_weight = getweight(weight_init(max_pos + 2 #= following HGF Bart's hack =#, dims, factor),
                           Layers.Embed, state_dict, joinname(prefix, "embed_positions.weight"))
    return Layers.FixedLenPositionEmbed(pos_weight)
end

function load_model(_type::Type{<:HGFBartPreTrainedModel}, ::Type{CompositeEmbedding}, cfg, state_dict, prefix)
    dims = cfg[:d_model]
    token_embed = load_model(_type, Layers.Embed, cfg, state_dict, joinname(prefix, "shared"))
    enc_pos = load_model(_type, Layers.FixedLenPositionEmbed, cfg, state_dict, joinname(prefix, "encoder"))
    dec_pos = load_model(_type, Layers.FixedLenPositionEmbed, cfg, state_dict, joinname(prefix, "decoder"))
    enc_embed = CompositeEmbedding(token = token_embed, position = (.+, enc_pos, bart_pe_shift))
    dec_embed = CompositeEmbedding(token = token_embed, position = (.+, dec_pos, bart_pe_shift))
    enc_ln = _load_layernorm(state_dict, joinname(prefix, "encoder.layernorm_embedding"), dims)
    dec_ln = _load_layernorm(state_dict, joinname(prefix, "decoder.layernorm_embedding"), dims)
    return Layers.Parallel{(:encoder_input, :decoder_input)}((
        Layers.Chain(enc_embed, enc_ln),
        Layers.Chain(dec_embed, dec_ln)))
end

load_model(::Type{<:HGFBartPreTrainedModel}, ::Type{Layers.LayerNorm}, cfg, state_dict, prefix) =
    _load_layernorm(state_dict, prefix, cfg[:d_model])

function load_model(
    ::Type{<:HGFBartPreTrainedModel},
    ::Type{<:SelfAttention{A}},
    cfg, state_dict, prefix
) where {A <: Union{MultiheadQKVAttenOp, CausalMultiheadQKVAttenOp}}
    dims = cfg[:d_model]
    head = cfg[A <: CausalMultiheadQKVAttenOp ? :encoder_attention_heads : :decoder_attention_heads]
    p = Float64(cfg[:attention_dropout]); p = iszero(p) ? nothing : p
    return_score = cfg[:output_attentions]
    factor = Float32(cfg[:init_std])
    qkv_proj = Layers.Fork(
        _load_dense(state_dict, joinname(prefix, "q_proj"), dims, dims, factor, true),
        _load_dense(state_dict, joinname(prefix, "k_proj"), dims, dims, factor, true),
        _load_dense(state_dict, joinname(prefix, "v_proj"), dims, dims, factor, true),
    )
    o_proj = _load_dense(state_dict, joinname(prefix, "out_proj"), dims, dims, factor, true)
    if A <: CausalMultiheadQKVAttenOp
        op = CausalMultiheadQKVAttenOp(head, p)
    else
        op = MultiheadQKVAttenOp(head, p)
    end
    return_score && (op = WithScore(op))
    return SelfAttention(op, qkv_proj, o_proj)
end
function load_model(::Type{<:HGFBartPreTrainedModel}, ::Type{Layers.CrossAttention}, cfg, state_dict, prefix)
    head, dims = cfg[:decoder_attention_heads], cfg[:d_model]
    p = Float64(cfg[:attention_dropout]); p = iszero(p) ? nothing : p
    return_score = cfg[:output_attentions]
    factor = Float32(cfg[:init_std])
    q_proj = _load_dense(state_dict, joinname(prefix, "q_proj"), dims, dims, factor, true)
    kv_proj = Layers.Fork(
        _load_dense(state_dict, joinname(prefix, "k_proj"), dims, dims, factor, true),
        _load_dense(state_dict, joinname(prefix, "v_proj"), dims, dims, factor, true),
    )
    o_proj = _load_dense(state_dict, joinname(prefix, "out_proj"), dims, dims, factor, true)
    op = MultiheadQKVAttenOp(head, p)
    return_score && (op = WithScore(op))
    return CrossAttention(op, q_proj, kv_proj, o_proj)
end

function _load_bart_ffn(state_dict, prefix, dims, ff_dims, factor, p, act)
    fc1 = _load_dense(state_dict, joinname(prefix, "fc1"), dims, ff_dims, factor, true, act)
    fc2 = _load_dense(state_dict, joinname(prefix, "fc2"), ff_dims, dims, factor, true)
    return Layers.Chain(Layers.DropoutLayer(fc1, p), fc2)
end

function load_model(_type::Type{<:HGFBartPreTrainedModel}, ::Type{<:TransformerBlock}, cfg, state_dict, prefix)
    dims = cfg[:d_model]
    ff_dims = cfg[:encoder_ffn_dim]
    factor = Float32(cfg[:init_std])
    act_p = Float64(cfg[:activation_dropout]); act_p = iszero(act_p) ? nothing : act_p
    act = ACT2FN[Symbol(cfg[:activation_function])]
    n = cfg[:encoder_layers]
    p = Float64(cfg[:dropout]); p = iszero(p) ? nothing : p
    collect_output = cfg[:output_attentions] || cfg[:output_hidden_states]
    blocks = []
    for i = 1:n
        lprefix = joinname(prefix, :layers, i-1)
        sa = load_model(_type, SelfAttention{MultiheadQKVAttenOp}, cfg, state_dict, joinname(lprefix, "self_attn"))
        sa_ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(lprefix, "self_attn_layer_norm"))
        sa = Layers.PostNormResidual(Layers.DropoutLayer(sa, p), sa_ln)
        ff = _load_bart_ffn(state_dict, lprefix, dims, ff_dims, factor, act_p, act)
        ff_ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(lprefix, "final_layer_norm"))
        ff = Layers.PostNormResidual(Layers.DropoutLayer(ff, p), ff_ln)
        block = TransformerBlock(sa, ff)
        push!(blocks, block)
    end
    collect_f = collect_output ? Layers.collect_outputs : nothing
    trf = Transformer(Tuple(blocks), collect_f)
    return trf
end

function load_model(_type::Type{<:HGFBartPreTrainedModel}, ::Type{<:TransformerDecoderBlock}, cfg, state_dict, prefix)
    dims = cfg[:d_model]
    ff_dims = cfg[:decoder_ffn_dim]
    factor = Float32(cfg[:init_std])
    act_p = Float64(cfg[:activation_dropout]); act_p = iszero(act_p) ? nothing : act_p
    act = ACT2FN[Symbol(cfg[:activation_function])]
    n = cfg[:decoder_layers]
    p = Float64(cfg[:dropout]); p = iszero(p) ? nothing : p
    collect_output = cfg[:output_attentions] || cfg[:output_hidden_states]
    blocks = []
    for i = 1:n
        lprefix = joinname(prefix, :layers, i-1)
        sa = load_model(_type, SelfAttention{CausalMultiheadQKVAttenOp}, cfg, state_dict, joinname(lprefix, "self_attn"))
        sa_ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(lprefix, "self_attn_layer_norm"))
        sa = Layers.PostNormResidual(Layers.DropoutLayer(sa, p), sa_ln)
        ca = load_model(_type, CrossAttention, cfg, state_dict, joinname(lprefix, "encoder_attn"))
        ca_ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(lprefix, "encoder_attn_layer_norm"))
        ca = Layers.PostNormResidual(Layers.DropoutLayer(ca, p), ca_ln)
        ff = _load_bart_ffn(state_dict, lprefix, dims, ff_dims, factor, act_p, act)
        ff_ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(lprefix, "final_layer_norm"))
        ff = Layers.PostNormResidual(Layers.DropoutLayer(ff, p), ff_ln)
        block = TransformerDecoderBlock(sa, ca, ff)
        push!(blocks, block)
    end
    collect_f = collect_output ? Layers.collect_outputs : nothing
    trf = Transformer(Tuple(blocks), collect_f)
    return trf
end

function get_state_dict(m::HGFBartModel, state_dict, prefix)
    get_state_dict(HGFBartModel, m.embed, state_dict, prefix)
    get_state_dict(HGFBartModel, m.seq2seq, state_dict, prefix)
    return state_dict
end

function get_state_dict(p::Type{<:HGFBartPreTrainedModel}, m::Layers.Parallel, state_dict, prefix)
    # m.layer.encoder_input[1].token === m.layer.decoder_input[1].token for BartModel
    get_state_dict(p, m.layer.encoder_input[1].token, state_dict, joinname(prefix, "shared"))
    get_state_dict(p, m.layer.encoder_input, state_dict, joinname(prefix, "encoder"))
    get_state_dict(p, m.layer.decoder_input, state_dict, joinname(prefix, "decoder"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFBartPreTrainedModel}, m::Layers.Chain{<:Tuple{CompositeEmbedding, Any}}, state_dict, prefix)
    get_state_dict(p, m[1].token, state_dict, joinname(prefix, "embed_tokens"))
    get_state_dict(p, m[1].position.embed, state_dict, joinname(prefix, "embed_positions"))
    get_state_dict(p, m[2], state_dict, joinname(prefix, "layernorm_embedding"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFBartPreTrainedModel}, m::Seq2Seq, state_dict, prefix)
    get_state_dict(p, m.encoder, state_dict, joinname(prefix, "encoder"))
    get_state_dict(p, m.decoder, state_dict, joinname(prefix, "decoder"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFBartPreTrainedModel}, m::Transformer, state_dict, prefix)
    for (i, t) in enumerate(m.blocks)
        get_state_dict(p, t, state_dict, joinname(prefix, "layers", i-1))
    end
    return state_dict
end

function get_state_dict(p::Type{<:HGFBartPreTrainedModel}, m::Layers.Chain{<:Tuple{Any, Layers.Dense}},
                        state_dict, prefix)
    get_state_dict(p, m[1], state_dict, joinname(prefix, "fc1"))
    get_state_dict(p, m[2], state_dict, joinname(prefix, "fc2"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFBartPreTrainedModel}, m::SelfAttention, state_dict, prefix)
    get_state_dict(p, m.qkv_proj.layers[1], state_dict, joinname(prefix, "q_proj"))
    get_state_dict(p, m.qkv_proj.layers[2], state_dict, joinname(prefix, "k_proj"))
    get_state_dict(p, m.qkv_proj.layers[3], state_dict, joinname(prefix, "v_proj"))
    get_state_dict(p, m.o_proj, state_dict, joinname(prefix, "out_proj"))
    return state_dict
end
function get_state_dict(p::Type{<:HGFBartPreTrainedModel}, m::CrossAttention, state_dict, prefix)
    get_state_dict(p, m.q_proj, state_dict, joinname(prefix, "q_proj"))
    get_state_dict(p, m.kv_proj.layers[1], state_dict, joinname(prefix, "k_proj"))
    get_state_dict(p, m.kv_proj.layers[2], state_dict, joinname(prefix, "v_proj"))
    get_state_dict(p, m.o_proj, state_dict, joinname(prefix, "out_proj"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFBartPreTrainedModel}, m::TransformerBlock, state_dict, prefix)
    get_state_dict(p, m.attention.layer, state_dict, joinname(prefix, "self_attn"))
    get_state_dict(p, m.attention.norm, state_dict, joinname(prefix, "self_attn_layer_norm"))
    get_state_dict(p, m.feedforward.layer, state_dict, prefix)
    get_state_dict(p, m.feedforward.norm, state_dict, joinname(prefix, "final_layer_norm"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFBartPreTrainedModel}, m::TransformerDecoderBlock, state_dict, prefix)
    get_state_dict(p, m.attention.layer, state_dict, joinname(prefix, "self_attn"))
    get_state_dict(p, m.attention.norm, state_dict, joinname(prefix, "self_attn_layer_norm"))
    get_state_dict(p, m.crossattention.layer, state_dict, joinname(prefix, "encoder_attn"))
    get_state_dict(p, m.crossattention.norm, state_dict, joinname(prefix, "encoder_attn_layer_norm"))
    get_state_dict(p, m.feedforward.layer, state_dict, prefix)
    get_state_dict(p, m.feedforward.norm, state_dict, joinname(prefix, "final_layer_norm"))
    return state_dict
end
