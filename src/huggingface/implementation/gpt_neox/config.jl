@hgfcfg :gpt_neox struct HGFGPTNeoXConfig
    vocab_size::Int = 50432
    hidden_size::Int = 6144
    num_hidden_layers::Int = 44
    num_attention_heads::Int = 64
    intermediate_size::Int = 24576
    hidden_act::String = "gelu"
    rotary_pct::Float64 = 0.25
    rotary_emb_base::Int = 10000
    classifier_dropout::Float64 = 0.1
    max_position_embeddings::Int = 2048
    initializer_range::Float64 = 0.02
    layer_norm_eps::Float64 = 1e-5
    use_cache::Bool = true
    bos_token_id::Int = 0
    eos_token_id::Int = 2
    tie_word_embeddings::Bool = false
    use_parallel_residual::Bool = true
end
