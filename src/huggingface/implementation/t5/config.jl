@hgfcfg :t5 struct HGFT5Config
    vocab_size::Int = 32128
    [d_model, hidden_size]::Int = 512
    d_kv::Int = 64
    d_ff::Int = 2048
    [num_layers, num_hidden_layers]::Int = 6
    num_decoder_layers::Int = 6
    [num_heads, num_attention_heads]::Int = 8
    relative_attention_num_buckets::Int = 32
    relative_attention_max_distance::Int = 128
    dropout_rate::Float64 = 0.1
    layer_norm_epsilon::Float32 = 1e-6
    initializer_factor::Float32 = 1.0
    feed_forward_proj::String = "relu"
    is_encoder_decoder::Bool = true
    use_cache::Bool = true
    pad_token_id::Int = 0
    eos_token_id::Int = 1
    dense_act_fn::Nothing = nothing
    is_gated_act::Bool = false
end

function HGFConfig{:t5}(cfg, overwrite)
    if !haskey(cfg, :num_decoder_layers) && haskey(cfg, :num_layers)
        overwrite[:num_decoder_layers] = cfg[:num_layers]
    end
    if haskey(cfg, :feed_forward_proj)
        feed_forward_proj = cfg.feed_forward_proj
    else
        feed_forward_proj = "relu"
        overwrite[:feed_forward_proj] = feed_forward_proj
    end
    act_info = split(feed_forward_proj, '-')
    overwrite[:dense_act_fn] = String(last(act_info))
    overwrite[:is_gated_act] = first(act_info) == "gated"
    return HGFConfig(:t5, cfg, overwrite)
end
