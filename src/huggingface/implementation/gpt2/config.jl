@hgfcfg :gpt2 struct HGFGPT2Config
    vocab_size::Int = 50257
    [n_positions, max_position_embeddings]::Int = 1024
    n_ctx::Int = 1024
    [n_embd, hidden_size]::Int = 768
    [n_layer, num_hidden_layers]::Int = 12
    [n_head, num_attention_heads]::Int = 12
    n_inner::Union{Nothing, Int} = nothing
    activation_function::String = "gelu_new"
    resid_pdrop::Float64 = 0.1
    embd_pdrop::Float64 = 0.1
    attn_pdrop::Float64 = 0.1
    layer_norm_epsilon::Float32 = 1e-5
    initializer_range::Float32 = 0.02
    summary_type::String = "cls_index"
    summary_use_proj::Bool = true
    summary_activation::Union{Nothing, String} = nothing
    summary_proj_to_labels::Bool = true
    summary_first_dropout::Float64 = 0.1
    bos_token_id::Int = 50256
    eos_token_id::Int = 50256
end
