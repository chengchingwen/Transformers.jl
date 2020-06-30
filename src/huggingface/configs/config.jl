using JSON

abstract type AbstractHGFConfig <: AbstractDict{Symbol, Any} end

Base.length(cfg::AbstractHGFConfig) = fieldcount(typeof(cfg))
Base.isempty(::AbstractHGFConfig) = false
Base.getindex(cfg::AbstractHGFConfig, k::Symbol) = getfield(cfg, k)
Base.iterate(cfg::AbstractHGFConfig) = iterate(cfg, 1)
Base.iterate(cfg::AbstractHGFConfig, i) = i <= length(cfg) ? (fieldname(HGFBertConfig, i)=> getfield(cfg, i), i+1) : nothing

function load_config(model_name)
  cfg = JSON.parsefile(get_registered_config_path(model_name); dicttype=Dict{Symbol, Any})
  model_type = Symbol(cfg[:model_type])
  return load_config(Val(model_type), cfg)
end

config_type(::Val{model_type}) where {model_type} = error("Unknown model type: $model_type")

function load_config(model_type::Val, cfg)
  cfg_type = config_type(model_type)
  ks = filter(fieldnames(cfg_type)) do k
    haskey(cfg, k)
  end
  vs = map(ks) do k
    cfg[k]
  end
  return cfg_type(; NamedTuple{ks}(vs)...)
end

include("./config_bert.jl")
