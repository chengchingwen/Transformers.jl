@hgfcfg :gptj struct HGFGPTJConfig
    vocab_size::Int = 50400
    [n_positions, max_position_embeddings]::Int = 2048
    [n_embd, hidden_size]::Int = 4096
    [n_layer, num_hidden_layers]::Int = 28
    [n_head, num_attention_heads]::Int = 16
    n_inner::Union{Nothing, Int} = nothing
    activation_function::String = "gelu_new"
    resid_pdrop::Float64 = 0.0
    embd_pdrop::Float64 = 0.0
    attn_pdrop::Float64 = 0.0
    layer_norm_epsilon::Float32 = 1e-5
    initializer_range::Float32 = 0.02
    scale_attn_weights::Bool = true
    use_cache::Bool = true
    bos_token_id::Int = 50256
    eos_token_id::Int = 50256
    tie_word_embeddings::Bool = false
end
