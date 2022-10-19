@cfgdef struct HGFT5Config <: HGFConfig
    vocab_size::Int = 32128
    d_model::Int = 512
    d_kv::Int = 64
    d_ff::Int = 2048
    num_layers::Int = 6
    num_decoder_layers::Int = 6
    num_heads::Int = 8
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
end

config_type(::Val{:t5}) = HGFT5Config

function load_config(::Type{HGFT5Config}, cfg)
    if !haskey(cfg, :num_decoder_layers) && haskey(cfg, :num_layers)
        cfg[:num_decoder_layers] = cfg[:num_layers]
    end
    return HGFT5Config(; cfg...)
end
