@hgfcfg :gpt_neo struct HGFGPTNeoConfig
    vocab_size::Int = 50257
    max_position_embeddings::Int = 2048
    hidden_size::Int = 2048
    [num_layers, num_hidden_layers]::Int = 24
    attention_types::Tuple{Tuple{NTuple{2, String}, Int}} =((("global", "local"), 12),)
    [num_heads, num_attention_heads]::Int = 16
    intermediate_size::Nothing = nothing
    window_size::Int = 256
    activation_function::String = "gelu_new"
    resid_dropout::Float64 = 0.0
    embed_dropout::Float64 = 0.0
    attention_dropout::Float64 = 0.0
    layer_norm_epsilon::Float64 = 1e-5
    initializer_range::Float64 = 0.02
    summary_type::String = "cls_index"
    summary_use_proj::Bool = true
    summary_activation::Nothing = nothing
    summary_proj_to_labels::Bool = true
    summary_first_dropout::Float64 = 0.1
    use_cache::Bool = true
    bos_token_id::Int = 50256
    eos_token_id::Int = 50256
end

function HGFConfig{:gpt_neo}(cfg, overwrite)
    if !haskey(cfg, :attention_layers)
        attention_types = get(cfg, :attention_types, ((("global", "local"), 12),))
        attentions = String[]
        for item in attention_types
            for _ in 1:(item[2])
                append!(attentions, item[1])
            end
        end
        overwrite[:attention_layers] = Tuple(attentions)
    end
    return HGFConfig(:gpt_neo, cfg, overwrite)
end
