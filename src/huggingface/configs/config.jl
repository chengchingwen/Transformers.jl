using JSON3
using ValSplit
using HuggingFaceApi

include("default.jl")

abstract type AbstractHGFConfig <: AbstractDict{Symbol, Any} end

struct HGFConfig{name, C, E <: Union{Nothing, Dict{Symbol, Any}}} <: AbstractHGFConfig
    pretrain::C
    overwrite::E
    function HGFConfig{name, C, E}(pretrain::C, overwrite::E) where {name, C, E <: Union{Nothing, Dict{Symbol, Any}}}
        !isnothing(overwrite) && isempty(overwrite) && (overwrite = nothing)
        return new{name, C, typeof(overwrite)}(pretrain, overwrite)
    end
end
HGFConfig(name::Symbol, pretrain, overwrite) = HGFConfig{name, typeof(pretrain), typeof(overwrite)}(pretrain, overwrite)
HGFConfig{name}(pretrain, overwrite = nothing) where name = HGFConfig{name, typeof(pretrain), typeof(overwrite)}(pretrain, overwrite)
function HGFConfig(cfg::HGFConfig{name}; kws...) where name
    overwrite = deepcopy(getfield(cfg, :overwrite))
    isnothing(overwrite) && (overwrite = Dict{Symbol, Any}())
    for k in keys(kws)
        overwrite[k] = kws[k]
    end
    return HGFConfig{name}(getfield(cfg, :pretrain), overwrite)
end

function Base.getproperty(cfg::HGFConfig, sym::Symbol)
    pretrain = getfield(cfg, :pretrain)
    overwrite = getfield(cfg, :overwrite)
    !isnothing(overwrite) && haskey(overwrite, sym) && return overwrite[sym]
    haskey(pretrain, sym) && return pretrain[sym]
    return getproperty(getdefault(cfg), sym)
end

getdefault(cfg::HGFConfig) = DEFAULT_PRETRAIN_CONFIG

Base.getindex(cfg::HGFConfig, sym::String) = cfg[Symbol(sym)]
Base.getindex(cfg::HGFConfig, sym::Symbol) = getproperty(cfg, sym)
Base.length(cfg::HGFConfig) = length(keys(cfg))

function Base.haskey(cfg::HGFConfig, k::Symbol)
    overwrite = getfield(cfg, :overwrite)
    if isnothing(overwrite)
        return haskey(getfield(cfg, :pretrain), k)
    else
        return haskey(overwrite, k) || haskey(getfield(cfg, :pretrain), k)
    end
end

function Base.keys(cfg::HGFConfig)
    overwrite = getfield(cfg, :overwrite)
    pretrain = getfield(cfg, :pretrain)
    return isnothing(overwrite) ? Tuple(keys(pretrain)) : Tuple(union(keys(pretrain), keys(overwrite)))
end

function Base.get(cfg::HGFConfig, k::Symbol, v)
    overwrite = getfield(cfg, :overwrite)
    pretrain = getfield(cfg, :pretrain)
    !isnothing(overwrite) && haskey(overwrite, k) && return overwrite[k]
    haskey(pretrain, k) && return pretrain[k]
    return v
end

function Base.iterate(cfg::HGFConfig, state = keys(cfg))
    isempty(state) && return nothing
    k = state[1]
    v = cfg[k]
    return k=>v, Base.tail(state)
end

Base.propertynames(cfg::HGFConfig) = keys(cfg)

getconfigname(::HGFConfig{name}) where name = name

include("auto.jl")
