@hgfcfg :bart struct HGFBartConfig
    vocab_size::Int = 50265
    max_position_embeddings::Int = 1024
    encoder_layers::Int = 12
    encoder_ffn_dim::Int = 4096
    [encoder_attention_heads, num_attention_heads]::Int = 16
    decoder_layers::Int = 12
    decoder_ffn_dim::Int = 4096
    decoder_attention_heads::Int = 16
    encoder_layerdrop::Float64 = 0.0
    decoder_layerdrop::Float64 = 0.0
    activation_function::String = "gelu"
    [d_model, hidden_size]::Int = 1024
    dropout::Float64 = 0.1
    attention_dropout::Float64 = 0.0
    activation_dropout::Float64 = 0.0
    init_std::Float32 = 0.02
    classifier_dropout::Float64 = 0.0
    scale_embedding::Bool = false
    use_cache::Bool = true
    num_labels::Int = 3
    pad_token_id::Int = 1
    bos_token_id::Int = 0
    eos_token_id::Int = 2
    is_encoder_decoder::Int = true
    decoder_start_token_id::Int = 2
    forced_eos_token_id::Int = 2
end
