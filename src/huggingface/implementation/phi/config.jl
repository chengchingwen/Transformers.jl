@hgfcfg :phi struct HGFPhiConfig
    vocab_size::Int = 51200
    hidden_size::Int = 2048
    intermediate_size::Int = 8192
    num_hidden_layers::Int = 24
    num_attention_heads::Int = 32
    num_key_value_heads::Nothing = nothing
    resid_pdrop::Float64 = 0.0
    embd_pdrop::Float64 = 0.0
    attention_dropout::Float64 = 0.0
    hidden_act::String = "gelu_new"
    max_position_embeddings::Int = 2048
    initializer_range::Float64 = 0.02
    layer_norm_eps::Float64 = 1e-5
    use_cache::Bool = true
    tie_word_embeddings::Bool = false
    rope_theta::Int = 10000
    rope_scaling::Nothing = nothing
    partial_rotary_factor::Float64 = 0.5
    qk_layernorm::Bool = false
    bos_token_id::Int = 1
    eos_token_id::Int = 2
end

function HGFConfig{:phi}(cfg, overwrite)
    if !haskey(cfg, :num_key_value_heads)
        overwrite[:num_key_value_heads] = get(cfg, :num_attention_heads, 32)
    end
    return HGFConfig(:phi, cfg, overwrite)
end
