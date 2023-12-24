@hgfcfg var"openai-gpt" struct HGFOpenAIGPTConfig
    vocab_size::Int = 40478
    [n_positions, max_position_embeddings]::Int = 512
    [n_embd, hidden_size]::Int = 768
    [n_layer, num_hidden_layers]::Int = 12
    [n_head, num_attention_heads]::Int = 12
    afn::String = "gelu"
    resid_pdrop::Float64 = 0.1
    embd_pdrop::Float64 = 0.1
    attn_pdrop::Float64 = 0.1
    layer_norm_epsilon::Float32 = 1e-5
    initializer_range::Float32 = 0.02
    predict_special_tokens::Bool = true
    summary_type::String = "cls_index"
    summary_use_proj::Bool = true
    summary_activation::Union{String, Nothing} = nothing
    summary_proj_to_labels::Bool = true
    summary_first_dropout::Float64 = 0.1
end
