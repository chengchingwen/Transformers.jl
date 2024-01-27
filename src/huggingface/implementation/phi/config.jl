@defaultdef :phi struct HGFPhiConfigDefault
    vocab_size::Int = 51200
    hidden_size::Int = 2048
    intermediate_size::Int = 8192
    num_hidden_layers::Int = 24
    num_attention_heads::Int = 32
    num_key_value_heads::Nothing = nothing
    hidden_act::String = "gelu_new"
    max_position_embeddings::Int = 2048
    initializer_range::Float64 = 0.02
    layer_norm_eps::Float64 = 1e-5
    use_cache::Bool = true
    pad_token_id::Nothing = nothing
    bos_token_id::Nothing = nothing
    eos_token_id::Nothing = nothing
    pretraining_tp::Int = 1
    embd_pdrop::Int = 0
    tie_word_embeddings::Bool = false
    rope_scaling::Nothing = nothing
    partial_rotary_factor::Float64 = 0.5
    qk_layernorm::Bool = false
    resid_pdrop::Float64 = 0.0
    rope_theta::Int = 10000
end