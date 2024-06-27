using ..Layers
using ..Layers: CompositeEmbedding, SelfAttention, CausalMultiheadQKVAttenOp, WithOptArg, WithArg
using Functors
using Static

using NeuralAttentionlib
using NeuralAttentionlib: WithScore, l2norm

@hgfdef CLIP (
    TextModel => (embed, encoder, pooler),
    VisionModel => (embed, encoder, pooler),
    Model => begin
      text_output = model.text_model(nt.text_input)
      vision_output = model.vision_model(nt.vision_input)
      nt2 = merge(Base.structdiff(nt, NamedTuple{(:text_input, :vision_input)}),
                  (embeddings = (text = l2norm(text_output.pooled), vision = l2norm(vision_output.pooled)),
                   text_output = text_output, vision_output = vision_output))
      logits = clip_cosine_similarity(nt2.embeddings.text, nt2.embeddings.vision, model.logit_scale)
      return merge(nt2, (logits = logits,))
    end,
    TextModelWithProjection => (embed, encoder, pooler),
    VisionModelWithProjection => (embed, encoder, pooler),
    # ForImageClassification,
)

# clip does not follow the common model structure
isbasemodel(::Type{<:HGFPreTrained{:clip}}) = true
isbasemodel(::Type{<:HGFPreTrained{:clip, :model}}) = true
basemodelkey(::Type{<:HGFPreTrained{:clip}}) = :clip

function load_model(_type::Type{HGFCLIPModel}, cfg, state_dict, prefix)
    text_model = load_model(_type, HGFCLIPTextModel, cfg, state_dict, prefix)
    vision_model = load_model(_type, HGFCLIPVisionModel, cfg, state_dict, prefix)
    logit_scale = getweight(Array, state_dict, joinname(prefix, "logit_scale")) do
        return fill(Float32(cfg.logit_scale_init_value))
    end
    return HGFCLIPModel(text_model, vision_model, logit_scale)
end

function load_model(_type::Type{HGFCLIPTextModelWithProjection}, cfg, state_dict, prefix)
    model = load_model(_type, HGFCLIPTextModel, cfg isa HGFCLIPConfig ? cfg[:text_config] : cfg, state_dict, prefix)
    return HGFCLIPTextModelWithProjection(model.embed, model.encoder, model.pooler)
end
function load_model(_type::Type{HGFCLIPVisionModelWithProjection}, cfg, state_dict, prefix)
    model = load_model(_type, HGFCLIPVisionModel, cfg isa HGFCLIPConfig ? cfg[:vision_config] : cfg, state_dict, prefix)
    return HGFCLIPVisionModelWithProjection(model.embed, model.encoder, model.pooler)
end

load_model(_type::Type{<:Union{HGFCLIPTextModel, HGFCLIPVisionModel}}, cfg, state_dict, prefix) =
    load_model(_type, _type, cfg, state_dict, prefix)
function load_model(_type::Type, ::Type{HGFCLIPTextModel}, cfg, state_dict, prefix)
    text_cfg = cfg isa HGFCLIPConfig ? cfg[:text_config] : cfg
    proj_dims = cfg[:projection_dim]
    dims = text_cfg[:hidden_size]
    ln_ϵ = text_cfg[:layer_norm_eps]
    factor = Float32(cfg[:initializer_factor]) / sqrt(dims)
    embed = load_model(HGFCLIPTextModel, CompositeEmbedding,
                       text_cfg, state_dict, joinname(prefix, "text_model.embeddings"))
    encoder = load_model(HGFCLIPTextModel, TransformerBlock,
                         text_cfg, state_dict, joinname(prefix, "text_model.encoder"))
    ln = _load_layernorm(state_dict, joinname(prefix, "text_model.final_layer_norm"), dims, ln_ϵ)
    if _type <: HGFCLIPTextModel
        proj = identity
    else
        proj = _load_dense(state_dict, joinname(prefix, "text_projection"), dims, proj_dims, factor, false)
    end
    return HGFCLIPTextModel(embed, Layers.Chain(encoder, ln), CLIPTextPooler(proj))
end
function load_model(_type::Type, ::Type{HGFCLIPVisionModel}, cfg, state_dict, prefix)
    vision_cfg = cfg isa HGFCLIPConfig ? cfg[:vision_config] : cfg
    proj_dims = cfg[:projection_dim]
    dims = vision_cfg[:hidden_size]
    ln_ϵ = vision_cfg[:layer_norm_eps]
    factor = Float32(cfg[:initializer_factor]) / sqrt(dims)
    embed = load_model(HGFCLIPVisionModel, CompositeEmbedding,
                       vision_cfg, state_dict, joinname(prefix, "vision_model.embeddings"))
    ln1 = _load_layernorm(state_dict, joinname(prefix, "vision_model.pre_layrnorm"), dims, ln_ϵ)
    encoder = load_model(HGFCLIPVisionModel, TransformerBlock,
                         vision_cfg, state_dict, joinname(prefix, "vision_model.encoder"))
    ln2 = _load_layernorm(state_dict, joinname(prefix, "vision_model.post_layernorm"), dims, ln_ϵ)
    if _type <: HGFCLIPVisionModel
        proj = identity
    else
        proj = _load_dense(state_dict, joinname(prefix, "visual_projection"), dims, proj_dims, factor, false)
    end
    return HGFCLIPVisionModel(Layers.Chain(embed, ln1), encoder, CLIPVisionPooler(ln2, proj))
end

function load_model(_type::Type{<:HGFCLIPTextModel}, ::Type{<:CompositeEmbedding}, cfg, state_dict, prefix)
    vocab_size, dims, max_pos = cfg[:vocab_size], cfg[:hidden_size], cfg[:max_position_embeddings]
    factor = Float32(cfg[:initializer_factor]) * 2f-2
    token_weight = getweight(weight_init(vocab_size, dims, factor), Layers.Embed,
                             state_dict, joinname(prefix, "token_embedding.weight"))
    pos_weight = getweight(weight_init(max_pos, dims, factor), Layers.Embed,
                           state_dict, joinname(prefix, "position_embedding.weight"))
    return CompositeEmbedding(token = Layers.Embed(token_weight), position = Layers.FixedLenPositionEmbed(pos_weight))
end
function load_model(_type::Type{<:HGFCLIPVisionModel}, ::Type{<:CompositeEmbedding}, cfg, state_dict, prefix)
    dims, filter, channel, image_size = cfg[:hidden_size], cfg[:patch_size], cfg[:num_channels], cfg[:image_size]
    num_patch = div(image_size, filter) ^ 2
    factor1 = Float32(cfg[:initializer_factor])
    factor2 = Float32(cfg[:initializer_range])
    factor = factor1 * factor2
    class_weight = getweight(bias_init(dims, factor1 / sqrt(dims)), Array,
                             state_dict, joinname(prefix, "class_embedding"))
    conv_weight = getweight(filter_init(filter, filter, channel, dims, factor), Flux.CrossCor,
                            state_dict, joinname(prefix, "patch_embedding.weight"))
    pos_weight = getweight(weight_init(num_patch, dims, factor), Layers.Embed,
                           state_dict, joinname(prefix, "position_embedding.weight"))
    return CompositeEmbedding(
        pixel = CLIPPixelEmbed(Flux.CrossCor(conv_weight; stride = filter), class_weight),
        position = Layers.FixedLenPositionEmbed(pos_weight))
end

function load_model(_type::Type{<:Union{HGFCLIPTextModel, HGFCLIPVisionModel}}, ::Type{<:SelfAttention}, cfg, state_dict, prefix)
    head, dims = cfg[:num_attention_heads], cfg[:hidden_size]
    @assert dims % head == 0 "The hidden size is not a multiple of the number of attention heads."
    p = cfg[:attention_dropout]; p = iszero(p) ? nothing : p
    return_score = cfg[:output_attentions]
    n = cfg[:num_hidden_layers]
    factor = Float32(cfg[:initializer_factor])
    afactor = factor / sqrt(dims * 2n)
    ofactor = factor / sqrt(dims)
    qkv_proj = Layers.Fork(
        _load_dense(state_dict, joinname(prefix, "q_proj"), dims, dims, afactor, true),
        _load_dense(state_dict, joinname(prefix, "k_proj"), dims, dims, afactor, true),
        _load_dense(state_dict, joinname(prefix, "v_proj"), dims, dims, afactor, true),
    )
    o_proj = _load_dense(state_dict, joinname(prefix, "out_proj"), dims, dims, ofactor, true)
    op = _type <: HGFCLIPTextModel ? CausalMultiheadQKVAttenOp(head, p) : MultiheadQKVAttenOp(head, p)
    return_score && (op = WithScore(op))
    return SelfAttention(op, qkv_proj, o_proj)
end

function load_model(
    _type::Type{<:HGFCLIPPreTrainedModel}, ::Type{<:Layers.Chain{<:Tuple{Layers.Dense, Layers.Dense}}},
    cfg, state_dict, prefix
)
    dims, ff_dims = cfg[:hidden_size], cfg[:intermediate_size]
    n = cfg[:num_hidden_layers]
    factor = Float32(cfg[:initializer_factor])
    ifactor = factor / sqrt(dims * 2n)
    ofactor = factor / sqrt(2dims)
    act = ACT2FN[Symbol(cfg[:hidden_act])]
    fc1 = _load_dense(state_dict, joinname(prefix, "fc1"), dims, ff_dims, ifactor, true, act)
    fc2 = _load_dense(state_dict, joinname(prefix, "fc2"), ff_dims, dims, ofactor, true)
    return Layers.Chain(fc1, fc2)
end

function load_model(_type::Type{<:HGFCLIPPreTrainedModel}, ::Type{<:TransformerBlock}, cfg, state_dict, prefix)
    dims = cfg[:hidden_size]
    n = cfg[:num_hidden_layers]
    collect_output = cfg[:output_attentions] || cfg[:output_hidden_states]
    ln_ϵ = cfg[:layer_norm_eps]
    cfg[:add_cross_attention] && load_error("Decoder Bert is not support.")
    blocks = []
    for i = 1:n
        lprefix = joinname(prefix, :layers, i-1)
        sa = load_model(_type, SelfAttention, cfg, state_dict, joinname(lprefix, "self_attn"))
        sa_ln = _load_layernorm(state_dict, joinname(lprefix, "layer_norm1"), dims, ln_ϵ)
        sa = Layers.PreNormResidual(sa, sa_ln)
        ff = load_model(_type, Layers.Chain{Tuple{Layers.Dense, Layers.Dense}}, cfg, state_dict, joinname(lprefix, "mlp"))
        ff_ln = _load_layernorm(state_dict, joinname(lprefix, "layer_norm2"), dims, ln_ϵ)
        ff = Layers.PreNormResidual(ff, ff_ln)
        block = TransformerBlock(sa, ff)
        push!(blocks, block)
    end
    collect_f = collect_output ? Layers.collect_outputs : nothing
    trf = Transformer(Tuple(blocks), collect_f)
    return trf
end

function get_state_dict(m::HGFCLIPModel, state_dict, prefix)
    get_state_dict(m.text_model, state_dict, prefix)
    get_state_dict(m.vision_model, state_dict, prefix)
    state_dict[joinname(prefix, "logit_scale")] = m.logit_scale
    return state_dict
end

function get_state_dict(m::Union{HGFCLIPTextModel, HGFCLIPTextModelWithProjection}, state_dict, prefix)
    get_state_dict(HGFCLIPTextModel, m.embed, state_dict, joinname(prefix, "text_model.embeddings"))
    get_state_dict(HGFCLIPTextModel, m.encoder[1], state_dict, joinname(prefix, "text_model.encoder"))
    get_state_dict(HGFCLIPTextModel, m.encoder[2], state_dict, joinname(prefix, "text_model.final_layer_norm"))
    if !isnothing(m.pooler.dense)
        get_state_dict(m.pooler.dense, state_dict, joinname(prefix, "text_projection"))
    end
    return state_dict
end
function get_state_dict(m::Union{HGFCLIPVisionModel, HGFCLIPVisionModelWithProjection}, state_dict, prefix)
    get_state_dict(HGFCLIPVisionModel, m.embed[1], state_dict, joinname(prefix, "vision_model.embeddings"))
    get_state_dict(HGFCLIPVisionModel, m.embed[2], state_dict, joinname(prefix, "vision_model.pre_layrnorm"))
    get_state_dict(HGFCLIPVisionModel, m.encoder, state_dict, joinname(prefix, "vision_model.encoder"))
    get_state_dict(HGFCLIPVisionModel, m.pooler.norm, state_dict, joinname(prefix, "vision_model.post_layernorm"))
    if !isnothing(m.pooler.dense)
        get_state_dict(m.pooler.dense, state_dict, joinname(prefix, "visual_projection"))
    end
    return state_dict
end

function get_state_dict(p::Type{<:HGFCLIPTextModel}, m::CompositeEmbedding, state_dict, prefix)
    get_state_dict(p, m.token, state_dict, joinname(prefix, "token_embedding"))
    get_state_dict(p, m.position.embed, state_dict, joinname(prefix, "position_embedding"))
    return state_dict
end
function get_state_dict(p::Type{<:HGFCLIPVisionModel}, m::CompositeEmbedding, state_dict, prefix)
    state_dict[joinname(prefix, "class_embedding")] = m.pixel.class_emb
    get_state_dict(p, m.pixel.conv, state_dict, joinname(prefix, "patch_embedding"))
    get_state_dict(p, m.position.embed, state_dict, joinname(prefix, "position_embedding"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFCLIPPreTrainedModel}, m::SelfAttention, state_dict, prefix)
    get_state_dict(p, m.qkv_proj.layers[1], state_dict, joinname(prefix, "q_proj"))
    get_state_dict(p, m.qkv_proj.layers[2], state_dict, joinname(prefix, "k_proj"))
    get_state_dict(p, m.qkv_proj.layers[3], state_dict, joinname(prefix, "v_proj"))
    get_state_dict(p, m.o_proj, state_dict, joinname(prefix, "out_proj"))
    return state_dict
end

function get_state_dict(
    p::Type{<:HGFCLIPPreTrainedModel}, m::Layers.Chain{<:Tuple{Layers.Dense, Layers.Dense}},
    state_dict, prefix
)
    get_state_dict(p, m[1], state_dict, joinname(prefix, "fc1"))
    get_state_dict(p, m[2], state_dict, joinname(prefix, "fc2"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFCLIPPreTrainedModel}, m::TransformerBlock, state_dict, prefix)
    get_state_dict(p, m.attention.layer, state_dict, joinname(prefix, "self_attn"))
    get_state_dict(p, m.attention.norm, state_dict, joinname(prefix, "layer_norm1"))
    get_state_dict(p, m.feedforward.layer, state_dict, joinname(prefix, "mlp"))
    get_state_dict(p, m.feedforward.norm, state_dict, joinname(prefix, "layer_norm2"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFCLIPPreTrainedModel}, m::Transformer, state_dict, prefix)
    for (i, t) in enumerate(m.blocks)
        get_state_dict(p, t, state_dict, joinname(prefix, :layers, i-1))
    end
    return state_dict
end
