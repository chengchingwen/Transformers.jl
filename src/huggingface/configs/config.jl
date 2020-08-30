using JSON
using MacroTools

abstract type AbstractHGFConfig <: AbstractDict{Symbol, Any} end
abstract type HGFPretrainedConfig <: AbstractHGFConfig end
abstract type HGFConfig <: AbstractHGFConfig end

Base.length(cfg::AbstractHGFConfig) = fieldcount(typeof(cfg))
Base.isempty(::AbstractHGFConfig) = false
Base.getindex(cfg::AbstractHGFConfig, k::Symbol) = getproperty(cfg, k)
Base.iterate(cfg::AbstractHGFConfig) = iterate(cfg, 1)
Base.iterate(cfg::AbstractHGFConfig, i) = i <= length(cfg) ? (fieldname(typeof(cfg), i) => getfield(cfg, i), i+1) : nothing

function load_config(model_name)
  cfg = JSON.parsefile(get_registered_config_path(model_name); dicttype=Dict{Symbol, Any})
  model_type = Symbol(cfg[:model_type])
  return load_config(Val(model_type), cfg)
end

config_type(::Val{model_type}) where {model_type} = error("Unknown model type: $model_type")

function load_config(model_type::Val, cfg)
  cfg_type = config_type(model_type)
  return cfg_type(; cfg...)
end

"""
  save_config(model_name, config; path=pwd(), config_name=DEFAULT_CONFIG_NAME)

save the `config` at `<path>/<model_name>/<config_name>`.
"""
function save_config(model_name, config; path=pwd(), config_name=DEFAULT_CONFIG_NAME)
  model_path = joinpath(path, model_name)
  !isdir(model_path) && error("$model_path is not a dir.")
  config_file = joinpath(model_path, config_name)
  open(config_file, "w+") do io
    JSON.print(io, config)
  end
  return config_file
end

struct HGFPretrainedConfigBase <: HGFPretrainedConfig
  # encoder-decoder
  is_encoder_decoder::Bool
  is_decoder::Bool
  add_cross_attention::Bool
  tie_encoder_decoder::Bool
  # sequence generation
  max_length::Int
  min_length::Int
  do_sample::Bool
  early_stopping::Bool
  num_beams::Int
  temeperature::Float64
  top_k::Int
  top_p::Float64
  repetition_penalty::Float64
  length_penalty::Float64
  no_repeat_ngram_size::Int
  bad_words_ids::Nothing
  num_return_sequence::Int
  chunk_size_feed_forward::Int
  # fine-tune task
  architectures::Nothing
  finetuning_task::Nothing
  id2label::Nothing
  label2id::Nothing
  num_labels::Int
  # tokenizer
  prefix::Nothing
  bos_token_id::Nothing
  pad_token_id::Nothing
  eos_token_id::Nothing
  decoder_start_token_id::Nothing

  HGFPretrainedConfigBase() = new(
    false, false, false, false,
    20, 0, false, false, 1, 1.0, 50, 1.0, 1.0, 1.0, 0, nothing, 1, 0,
    nothing, nothing, nothing, nothing, 2,
    nothing, nothing, nothing, nothing, nothing,
  )
end

struct UpdatedConfig{C <: HGFPretrainedConfig, V} <: HGFPretrainedConfig
  config::C
  name::Symbol
  value::V
end

update(cfg::HGFPretrainedConfigBase, name::Symbol, v) = UpdatedConfig(cfg, name, v)
function update(cfg::UpdatedConfig, name::Symbol, v)
  if _name(cfg) == name
    return UpdatedConfig(_config(cfg), name, v)
  else
    return UpdatedConfig(
      update(_config(cfg), name, v),
      _name(cfg), _value(cfg))
  end
end

_name(cfg::UpdatedConfig) = getfield(cfg, :name)
_value(cfg::UpdatedConfig) = getfield(cfg, :value)
_config(cfg::UpdatedConfig) = getfield(cfg, :config)
_base(cfg::UpdatedConfig) = (_base ∘ _config)(cfg)
_base(cfg::HGFPretrainedConfigBase) = cfg

Base.getproperty(cfg::UpdatedConfig, k::Symbol) = k == _name(cfg) ? _value(cfg) : Base.getproperty(_config(cfg), k)

_get_cfg(cfg::UpdatedConfig, k) = k == _name(cfg) ? cfg : _get_cfg(_config(cfg), k)

_current_pair(cfg::UpdatedConfig) = _name(cfg)=>_value(cfg)

_updated(cfg::UpdatedConfig) = tuple(_current_pair(cfg), _updated(_config(cfg))...)
_updated(cfg::HGFPretrainedConfigBase) = ()

Base.iterate(cfg::UpdatedConfig) = iterate(cfg, _name(cfg))
function Base.iterate(cfg::UpdatedConfig, i)
  isnothing(i) && return nothing
  cur = _get_cfg(cfg, i)
  p = _current_pair(cur)
  nc = _config(cur)
  if nc isa UpdatedConfig
    return p, _name(nc)
  else
    return p, nothing
  end
end

function pretrained_config(; kws...)
  global PRETRAIN_CONFIG
  p = PRETRAIN_CONFIG
	for (k, v) in kws
    p = update(p, k, v)
  end
  return p
end

_parent(cfg::HGFConfig) = getfield(cfg, :_parent)

Base.length(cfg::UpdatedConfig) = length(_updated(cfg))
function Base.summary(io::IO, x::UpdatedConfig)
  n = length(x)
  print(io, typeof(x).name)
  print(io, " with ", n, (n==1 ? " entry" : " entries"))
end

const PRETRAIN_CONFIG = HGFPretrainedConfigBase()

Base.length(cfg::HGFConfig) = fieldcount(typeof(cfg)) + length(_updated(_parent(cfg))) - 1
Base.iterate(cfg::HGFConfig) = iterate(cfg, 1)
function Base.iterate(cfg::HGFConfig, i::Int)
  n = fieldcount(typeof(cfg))
  if i < n
    return fieldname(typeof(cfg), i) => getfield(cfg, i), i+1
  else
    pcfg = _parent(cfg)
    if pcfg isa UpdatedConfig
      return iterate(pcfg)
    else
      return nothing
    end
  end
end

Base.iterate(cfg::HGFConfig, i::Union{Symbol, Nothing}) = iterate(_parent(cfg), i)

macro cfgdef(ex)
  @capture(ex, struct T_ fields__ end) || error("only accept struct definition like Base.@kwdef")
  fields_wo_default = map(fields) do fwv
    @capture(fwv, f_ = v_) ? f : fwv
  end
  field_names = map(fields_wo_default) do f
    @capture(f, fn_::ft_) ? fn : f
  end
  @capture(T, Tname_ <: Sname_) || (Tname = T)
  Tname = esc(Tname)
  param_ex = map(fields) do fwv
    @capture(fwv, f_ = v_) ? Expr(:kw, @capture(f, fn_::ft_) ? fn : f, esc(v)) : fwv
  end
  kws = gensym(:kws)
  push!(param_ex, :($(kws)...))
  param_ex = Expr(:parameters, param_ex...)

  return quote
    struct $(esc(T))
      $(esc.(fields_wo_default)...)
      _parent::HGFPretrainedConfig
    end

    function $(Tname)($(param_ex))
      p = pretrained_config(; $(kws)...)
      $(Tname)($(field_names...), p)
    end
  end
end

Base.getproperty(cfg::HGFConfig, k::Symbol) = hasfield(typeof(cfg), k) ? getfield(cfg, k) : getproperty(_parent(cfg), k)

include("./config_bert.jl")
include("./config_gpt2.jl")
include("./config_roberta.jl")
