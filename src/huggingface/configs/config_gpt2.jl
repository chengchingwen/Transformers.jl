@cfgdef struct HGFGPT2Config <: HGFConfig
  vocab_size::Int = 50257
  n_positions::Int = 1024
  n_ctx::Int = 1024
  n_embd::Int = 768
  n_layer::Int = 12
  n_head::Int = 12
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

config_type(::Val{:gpt2}) = HGFGPT2Config
