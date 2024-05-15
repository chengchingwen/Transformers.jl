@hgfcfg :llama struct HGFLlamaConfig
    vocab_size::Int = 32000
    hidden_size::Int = 4096
    intermediate_size::Int = 11008
    num_hidden_layers::Int = 32
    num_attention_heads::Int = 32
    num_key_value_heads::Nothing = nothing
    hidden_act::String = "silu"
    max_position_embeddings::Int = 2048
    initializer_range::Float64 = 0.02
    rms_norm_eps::Float64 = 1e-6
    use_cache::Bool = true
    pad_token_id::Nothing = nothing
    bos_token_id::Int = 1
    eos_token_id::Int = 2
    pretraining_tp::Int = 1
    tie_word_embeddings::Bool = false
    rope_theta::Float64 = 1e4
    rope_scaling::Nothing = nothing
    clean_up_tokenization_spaces::Bool = false
end
