using JSON3
using ValSplit
using HuggingFaceApi

include("default.jl")

abstract type AbstractHGFConfig <: AbstractDict{Symbol, Any} end

struct HGFConfig{name, C, E <: Union{Nothing, NamedTuple}} <: AbstractHGFConfig
    pretrain::C
    overwrite::E
    function HGFConfig{name, C}(pretrain::C, overwrite::Union{Nothing, NamedTuple}) where {name, C}
        return new{name, C, typeof(overwrite)}(pretrain, overwrite)
    end
    function HGFConfig{name, C}(pretrain::C, overwrite::AbstractDict{Symbol}) where {name, C}
        isempty(overwrite) && return HGFConfig{name, C}(pretrain, nothing)
        namemap = getnamemap(HGFConfig{name})
        if !isempty(aliases(namemap))
            rewrite = Dict{Symbol, Any}()
            for (k, val) in overwrite
                key = haskey(namemap, k) ? aliasof(namemap, k) : k
                haskey(rewrite, key) &&
                    error("Aliases of the same config entry is set at the same time: $(aliasgroup(namemap, key))")
                rewrite[key] = val
            end
            overwrite = rewrite
        end
        overwrite = NamedTuple(overwrite)
        return new{name, C, typeof(overwrite)}(pretrain, overwrite)
    end
end
@inline HGFConfig(name::Symbol, pretrain, overwrite) = HGFConfig{name, typeof(pretrain)}(pretrain, overwrite)
HGFConfig{name}(pretrain, overwrite = nothing) where name = HGFConfig(name, pretrain, overwrite)
function HGFConfig(cfg::HGFConfig{name}; kws...) where name
    overwrite = getfield(cfg, :overwrite)
    if isnothing(overwrite)
        overwrite = Dict{Symbol, Any}(kws)
    else
        namemap = getnamemap(HGFConfig{name})
        for k in keys(kws)
            if haskey(namemap, k)
                count(Base.Fix1(haskey, kws), aliasgroup(namemap, k)) > 1 &&
                    error("Aliases of the same config entry is set at the same time: $(aliasgroup(namemap, k))")
            end
        end
        overwrite = merge(overwrite, kws)
    end
    pretrain = getfield(cfg, :pretrain)
    return HGFConfig{name, typeof(pretrain)}(pretrain, overwrite)
end

getconfigname(::HGFConfig{name}) where name = name

getdefault(cfg::HGFConfig) = getdefault(typeof(cfg))
getdefault(::Type{<:HGFConfig}) = DEFAULT_PRETRAIN_CONFIG

getnamemap(cfg::HGFConfig) = getnamemap(typeof(cfg))
getnamemap(::Type{<:HGFConfig}) = getfield(DEFAULT_PRETRAIN_CONFIG, :namemap)

function Base.propertynames(cfg::HGFConfig)
    global DEFAULT_PRETRAIN_CONFIG
    namemap = getnamemap(cfg)
    overwrite = getfield(cfg, :overwrite)
    return union(
        isnothing(overwrite) ? () : keys(overwrite),
        keys(getfield(cfg, :pretrain)),
        keys(namemap),
        keys(DEFAULT_PRETRAIN_CONFIG))
end

function Base.hasproperty(cfg::HGFConfig, sym::Symbol)
    global DEFAULT_PRETRAIN_CONFIG
    namemap = getnamemap(cfg)
    haskey(namemap, sym) && return true
    overwrite = getfield(cfg, :overwrite)
    overwritten = isnothing(overwrite) ? false : haskey(overwrite, sym)
    return overwritten || haskey(getfield(cfg, :pretrain), sym) || haskey(DEFAULT_PRETRAIN_CONFIG, sym)
end

function Base.getproperty(cfg::HGFConfig, sym::Symbol)
    global DEFAULT_PRETRAIN_CONFIG
    namemap = getnamemap(cfg)
    haskey(namemap, sym) && (sym = aliasof(namemap, sym)::Symbol)
    overwrite = getfield(cfg, :overwrite)
    !isnothing(overwrite) && haskey(overwrite, sym) && return overwrite[sym]
    pretrain = getfield(cfg, :pretrain)
    haskey(pretrain, sym) && return pretrain[sym]
    for key in aliasgroup(namemap, sym)
        haskey(pretrain, key) && return pretrain[key]
    end
    default = getdefault(cfg)
    haskey(default, sym) && return default[sym]
    haskey(DEFAULT_PRETRAIN_CONFIG, sym) && return DEFAULT_PRETRAIN_CONFIG[sym]
    error("type HGFConfig{$(getconfigname(cfg))} has no field $sym")
end

Base.getindex(cfg::HGFConfig, sym::String) = cfg[Symbol(sym)]
Base.getindex(cfg::HGFConfig, sym::Symbol) = getproperty(cfg, sym)
Base.length(cfg::HGFConfig) = length(keys(cfg))

function Base.keys(cfg::HGFConfig)
    namemap = getnamemap(cfg)
    overwrite = getfield(cfg, :overwrite)
    pretrain = getfield(cfg, :pretrain)
    names = Set{Symbol}()
    for set in (pretrain, something(overwrite, ())), k in keys(set)
        haskey(namemap, k) && (k = aliasof(namemap, k)::Symbol)
        push!(names, k)
    end
    return Tuple(names)
end

Base.haskey(cfg::HGFConfig, k::String) = haskey(cfg, Symbol(k))
function Base.haskey(cfg::HGFConfig, k::Symbol)
    namemap = getnamemap(cfg)
    haskey(namemap, k) && (k = aliasof(namemap, k)::Symbol)
    overwrite = getfield(cfg, :overwrite)
    overwritten = isnothing(overwrite) ? false : haskey(overwrite, k)
    return overwritten || haskey(getfield(cfg, :pretrain), k)
end

Base.get(cfg::HGFConfig, k::String, v) = get(cfg, Symbol(k), v)
function Base.get(cfg::HGFConfig, k::Symbol, v)
    namemap = getnamemap(cfg)
    haskey(namemap, k) && (k = aliasof(namemap, k)::Symbol)
    overwrite = getfield(cfg, :overwrite)
    !isnothing(overwrite) && haskey(overwrite, k) && return overwrite[k]
    pretrain = getfield(cfg, :pretrain)
    return haskey(pretrain, k) ? pretrain[k] : v
end

function Base.iterate(cfg::HGFConfig, state = keys(cfg))
    isempty(state) && return nothing
    k = state[1]
    v = cfg[k]
    return k=>v, Base.tail(state)
end

function Base.summary(io::IO, cfg::HGFConfig)
    print(io, "HGFConfig{")
    show(io, getconfigname(cfg))
    print(io, "} with ")
    len = length(cfg)
    print(io, len)
    print(io, len == 1 ? " entry" : " entries")
end

include("auto.jl")

# api doc

"""
    HGFConfig{model_type} <: AbstractDict{Symbol, Any}

The type for holding the configuration for huggingface model `model_type`.

    HGFConfig(base_cfg::HGFConfig; kwargs...)

Return a new `HGFConfig` object for the same `model_type` with fields updated with `kwargs`.

# Example

```julia-repl
julia> bertcfg = load_config("bert-base-cased");

julia> bertcfg.num_labels
2

julia> mycfg = HuggingFace.HGFConfig(bertcfg; num_labels = 3);

julia> mycfg.num_labels
3

```

# Extended help

Each `HGFConfig` has a pre-defined set of type-dependent default field values and some field name aliases. For example,
 `(cfg::HGFConfig{:gpt2}).hidden_size` is an alias of `(cfg::HGFConfig{:gpt2}).n_embd`. Using `propertynames`,
 `hasproperty`, `getproperty`, `getindex` will access the default field values if the key is not present in the loaded
 configuration. On the other hand, using `length`, `keys`, `haskey`, `get`, `iterate` will not interact with the
 default values (while the name aliases still work).

```julia-repl
julia> fakegpt2cfg = HuggingFace.HGFConfig{:gpt2}((a=3,b=5))
Transformers.HuggingFace.HGFConfig{:gpt2, @NamedTuple{a::Int64, b::Int64}, Nothing} with 2 entries:
  :a => 3
  :b => 5

julia> myfakegpt2cfg = HuggingFace.HGFConfig(fakegpt2cfg; hidden_size = 7)
Transformers.HuggingFace.HGFConfig{:gpt2, @NamedTuple{a::Int64, b::Int64}, @NamedTuple{n_embd::Int64}} with 3 entries:
  :a      => 3
  :b      => 5
  :n_embd => 7

julia> myfakegpt2cfg[:hidden_size] == myfakegpt2cfg.hidden_size == myfakegpt2cfg.n_embd
true

julia> myfakegpt2cfg.n_layer
12

julia> get(myfakegpt2cfg, :n_layer, "NOT_FOUND")
"NOT_FOUND"

```
"""
HGFConfig
