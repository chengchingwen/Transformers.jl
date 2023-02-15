@defaultdef :gptj struct HGFGPTJConfigDefault
    vocab_size::Int = 50400
    n_positions::Int = 2048
    n_embd::Int = 4096
    n_layer::Int = 28
    n_head::Int = 16
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

const HGFGPTJConfig = HGFConfig{:gptj}
