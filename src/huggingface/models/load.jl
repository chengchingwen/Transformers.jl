import ..Layers

using Flux
using NNlib
using DataStructures: OrderedDict

using LinearAlgebra

struct FirstTokenPooler end
(m::FirstTokenPooler)(x) = selectdim(x, 2, 1)

abstract type HGFPreTrainedModel end
Layers.@fluxshow HGFPreTrainedModel

const ACT2FN = (
    gelu = gelu, gelu_new = gelu, quick_gelu = gelu,
    swish = swish, silu = swish,
    relu = relu,
    mish = mish,
    selu = selu,
)

joinname(prefix, name) = isempty(prefix) ? name : join((prefix, name), '.')
joinname(prefix, n1, n2...) = joinname(prefix, join((n1, n2...), '.'))

haskeystartswith(dict, prefix) = any(startswith("$prefix."), keys(dict))

zero_init(dims) = () -> zeros(Float32, dims)
one_init(dims) = () -> ones(Float32, dims)
function weight_init(din, dout, factor = true)
    function weight_init_f() # normal(mean = 0, std = factor)
        weight = randn(Float32, dout, din)
        if !isone(factor)
            weight .*= factor
        end
        return weight
    end
    return weight_init_f
end

collect32(x) = collect(Float32, x)

getweight(init, ::Type, ::Symbol) = init()
getweight(init,  x, sym::Symbol) = getproperty(x, sym)

getweight(init, ::Type{<:Array}, state_dict, name) = _getweight(collect32, init, state_dict, name)
getweight(init, ::Type{<:Array}, state_dict::OrderedDict{String}, name) = getweight(init, state_dict, name)
getweight(init, ::Type{<:Layers.Embed}, state_dict, name) = _getweight(collect32 ∘ adjoint, init, state_dict, name)
getweight(init, ::Type{<:Layers.Embed}, state_dict::OrderedDict{String}, name) = _getweight(adjoint, init, state_dict, name)

getweight(init, state_dict, name) = _getweight(identity, init, state_dict, name)
function _getweight(process, init, state_dict, name)
    if haskey(state_dict, name)
        weight = state_dict[name] |> process
    else
        @debug "$name not found, initialized."
        weight = init()
    end
    return weight
end

get_state_dict(_, m::Layers.Embed, state_dict, prefix) = get_state_dict(m, state_dict, prefix)
function get_state_dict(m::Layers.Embed, state_dict, prefix)
    state_dict[joinname(prefix, "weight")] = m.embeddings'
    return state_dict
end

get_state_dict(_, m::Layers.FixedLenPositionEmbed, state_dict, prefix) = get_state_dict(m, state_dict, prefix)
function get_state_dict(m::Layers.FixedLenPositionEmbed, state_dict, prefix)
    state_dict[joinname(prefix, "weight")] = m.embeddings'
    return state_dict
end

get_state_dict(_, m::Layers.Dense, state_dict, prefix) = get_state_dict(m, state_dict, prefix)
function get_state_dict(m::Layers.Dense, state_dict, prefix)
    state_dict[joinname(prefix, "weight")] = m.W
    !isnothing(m.b) && (state_dict[joinname(prefix, "bias")] = m.b)
    return state_dict
end

get_state_dict(_, m::Layers.LayerNorm, state_dict, prefix) = get_state_dict(m, state_dict, prefix)
function get_state_dict(m::Layers.LayerNorm, state_dict, prefix)
    state_dict[joinname(prefix, "weight")] = m.α
    state_dict[joinname(prefix, "bias")] = m.β
    return state_dict
end

get_state_dict(_, m::Layers.RMSLayerNorm, state_dict, prefix) = get_state_dict(m, state_dict, prefix)
function get_state_dict(m::Layers.RMSLayerNorm, state_dict, prefix)
    state_dict[joinname(prefix, "weight")] = m.α
    return state_dict
end

get_state_dict(p, m::Layers.RenameArgs, state_dict, prefix) = get_state_dict(p, m.layer, state_dict, prefix)
get_state_dict(p, m::Layers.Branch, state_dict, prefix) = get_state_dict(p, m.layer, state_dict, prefix)
get_state_dict(p, m::Layers.Parallel, state_dict, prefix) = get_state_dict(p, m.layer, state_dict, prefix)
get_state_dict(p, m::Layers.DropoutLayer, state_dict, prefix) = get_state_dict(p, m.layer, state_dict, prefix)

load_model(_type::Type, cfg) = load_model(_type, cfg, OrderedDict{String, Any}())
load_model(_type::Type, cfg, state_dict) = load_model(_type, cfg, state_dict, "")

get_state_dict(m) = get_state_dict(m, OrderedDict{String, Any}())
get_state_dict(m, state_dict) = get_state_dict(m, state_dict, "")
