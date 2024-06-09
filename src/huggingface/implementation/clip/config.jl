@hgfcfg :clip_text_model struct HGFCLIPTextConfig
    vocab_size::Int = 49408
    hidden_size::Int = 512
    intermediate_size::Int = 2048
    num_hidden_layers::Int = 12
    num_attention_heads::Int = 8
    max_position_embeddings::Int = 77
    hidden_act::String = "quick_gelu"
    layer_norm_eps::Float64 = 1e-5
    attention_dropout::Float64 = 0.0
    dropout::Float64 = 0.0
    initializer_range::Float64 = 0.02
    initializer_factor::Float64 = 1
end

@hgfcfg :clip_vision_model struct HGFCLIPVisionConfig
    hidden_size::Int = 768
    intermediate_size::Int = 3072
    num_hidden_layers::Int = 12
    num_attention_heads::Int = 12
    num_channels::Int = 3
    image_size::Int = 224
    patch_size::Int = 32
    hidden_act::String = "quick_gelu"
    layer_norm_eps::Float64 = 0.00001
    dropout::Float64 = 0.0
    attention_dropout::Float64 = 0.0
    initializer_range::Float64 = 0.02
    initializer_factor::Float64 = 1.0
end

@hgfcfg :clip struct HGFCLIPConfig
    text_config::HGFCLIPTextConfig = HGFCLIPTextConfig((;))
    vision_config::HGFCLIPVisionConfig = HGFCLIPVisionConfig((;))
    text_config_dict::Nothing = nothing
    vision_config_dict::Nothing = nothing
    projection_dim::Int = 512
    logit_scale_init_value::Float64 = 2.6592
end

function HGFConfig{:clip}(cfg, overwrite)
    text_config = HGFConfig{:clip_text_model}(
        haskey(cfg, :text_config) ? cfg.text_config :
        haskey(cfg, :text_config_dict) ? cfg.text_config_dict : (;))
    overwrite[:text_config] = text_config
    vision_config = HGFConfig{:clip_vision_model}(
        haskey(cfg, :vision_config) ? cfg.vision_config :
        haskey(cfg, :vision_config_dict) ? cfg.vision_config_dict : (;))
    overwrite[:vision_config] = vision_config
    return HGFConfig(:clip, cfg, overwrite)
end
