@cfgdef struct HGFCLIPTextConfig <: HGFConfig
    vocab_size::Int =  49408
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

@cfgdef struct HGFCLIPVisionConfig <: HGFConfig
    hidden_size::Int =768
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

@cfgdef struct HGFCLIPConfig <: HGFConfig
    text_config::HGFCLIPTextConfig
    vision_config::HGFCLIPVisionConfig
    # text_config_dict::Union{Dict, Nothing} = nothing
    # vision_config_dict::Dict{Dict, Nothing} = nothing
    projection_dim::Int = 512
    logit_scale_init_value::Float64 = 2.6592
end

config_type(::Val{:clip}) = HGFCLIPConfig

function load_config(::Type{HGFCLIPConfig}, cfg)
    text_config = get(cfg, :text_config, nothing)
    if isnothing(text_config)
        text_config_dict = get(cfg, :text_config_dict, (;))
    else
        text_config_dict = Dict{Symbol, Any}(Symbol(k) => v for (k, v) in text_config)
    end
    text_config = load_config(HGFCLIPTextConfig, text_config_dict)

    vision_config = get(cfg, :vision_config, nothing)
    if isnothing(vision_config)
        vision_config_dict = get(cfg, :vision_config_dict, (;))
    else
        vision_config_dict = Dict{Symbol, Any}(Symbol(k) => v for (k, v) in vision_config)
    end
    vision_config = load_config(HGFCLIPVisionConfig, vision_config_dict)

    kws = Dict{Symbol, Any}()
    for (k, v) in cfg
        if k == :text_config
            kws[:text_config] = text_config
        elseif k == :vision_config
            kws[:vision_config] = vision_config
        else
            kws[k] = v
        end
    end

    return HGFCLIPConfig(; kws...)
end
