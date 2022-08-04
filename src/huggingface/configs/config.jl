using JSON
using MacroTools
using ValSplit

using HuggingFaceApi

abstract type AbstractHGFConfig <: AbstractDict{Symbol, Any} end
abstract type HGFPretrainedConfig <: AbstractHGFConfig end
abstract type HGFConfig <: AbstractHGFConfig end

Base.length(cfg::AbstractHGFConfig) = fieldcount(typeof(cfg))
Base.isempty(::AbstractHGFConfig) = false
Base.getindex(cfg::AbstractHGFConfig, k::Symbol) = getproperty(cfg, k)
Base.iterate(cfg::AbstractHGFConfig) = iterate(cfg, 1)
Base.iterate(cfg::AbstractHGFConfig, i) = i <= length(cfg) ? (fieldname(typeof(cfg), i) => getfield(cfg, i), i+1) : nothing
Base.propertynames(cfg::AbstractHGFConfig) = keys(cfg)

_parent(cfg::HGFConfig) = getfield(cfg, :_parent)

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

function Base.getproperty(cfg::HGFConfig, k::Symbol)
    if hasfield(typeof(cfg), k)
        return getfield(cfg, k)
    else
        if k == :num_labels
            id2label = getproperty(cfg, :id2label)
            if !isnothing(id2label)
                return length(id2label)
            end
        end
        return getproperty(_parent(cfg), k)
    end
end

function Base.get(cfg::HGFConfig, k::Symbol, d)
    if hasfield(typeof(cfg), k)
        return getfield(cfg, k)
    else
        if k == :num_labels
            id2label = cfg.id2label
            if !isnothing(id2label)
                return length(id2label)
            end
        end
        return get(_parent(cfg), k, d)
    end
end

include("./auto.jl")
include("./pretrain.jl")
