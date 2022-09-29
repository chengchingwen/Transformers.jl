@cfgdef struct HGFOpenAIGPTConfig <: HGFConfig
    vocab_size::Int = 40478
    n_positions::Int = 512
    n_embd::Int = 768
    n_layer::Int = 12
    n_head::Int = 12
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

config_type(::Val{Symbol("openai-gpt")}) = HGFOpenAIGPTConfig
