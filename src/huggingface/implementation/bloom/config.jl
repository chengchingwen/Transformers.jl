@hgfcfg :bloom struct HGFBloomConfig
    vocab_size::Int = 250880
    hidden_size::Int = 64
    [n_layer, num_hidden_layers]::Int = 2
    [n_head, num_attention_heads]::Int = 8
    layer_norm_epsilon::Float64 = 1e-5
    initializer_range::Float64 = 0.02
    use_cache::Bool = true
    bos_token_id::Int = 1
    eos_token_id::Int = 2
    apply_residual_connection_post_layernorm::Bool = false
    hidden_dropout::Float64 = 0.0
    attention_dropout::Float64 = 0.0
    pretraining_tp::Int = 1
    slow_but_exact::Bool = false
    clean_up_tokenization_spaces::Bool = false
end
