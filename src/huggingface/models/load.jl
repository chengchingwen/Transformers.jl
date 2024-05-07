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
        state = state_dict[name]
        if Pickle.Torch.islazy(state)
            lazystate = state
            if Pickle.Torch.isloaded(lazystate)
                weight = lazystate.data
            else
                state = lazystate()
                weight = process(state)
                lazystate.data = weight
            end
        else
            weight = process(state)
        end
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

get_state_dict(_, m::Layers.EmbedDecoder, state_dict, prefix) = get_state_dict(m, state_dict, prefix)
function get_state_dict(m::Layers.EmbedDecoder, state_dict, prefix)
    if !isnothing(m.bias)
        state_dict[joinname(prefix, "bias")] = m.bias
    end
    get_state_dict(m.embed, state_dict, prefix)
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


_load_embed(state_dict, prefix, vocab_size, dims, factor, pad_idx0 = nothing) =
    _load_embed(state_dict, prefix, weight_init(vocab_size, dims, factor), pad_idx0)
function _load_embed(state_dict, prefix, w_init, pad_idx0 = nothing)
    embedding = getweight(Layers.Embed, state_dict, joinname(prefix, "weight")) do
        weight = w_init()
        if !isnothing(pad_idx0)
            weight[:, pad_idx0 + 1] .= 0
        end
        return weight
    end
    return Layers.Embed(embedding)
end

function _load_layernorm(state_dict, prefix, dims, ln_ϵ)
    old_weight_name = joinname(prefix, "gamma")
    old_bias_name = joinname(prefix, "beta")
    weight_name = haskey(state_dict, old_weight_name) ? old_weight_name : joinname(prefix, "weight")
    bias_name = haskey(state_dict, old_bias_name) ? old_bias_name : joinname(prefix, "bias")
    ln_weight = getweight(one_init(dims), Array, state_dict, weight_name)
    ln_bias = getweight(zero_init(dims), Array, state_dict, bias_name)
    return Layers.LayerNorm(ln_weight, ln_bias, ln_ϵ)
end

_load_dense(state_dict, prefix, din, dout, factor, bias, act = nothing) =
    _load_dense(state_dict, prefix, weight_init(din, dout, factor), bias ? zero_init(dout) : nothing, act)
function _load_dense(state_dict, prefix, w_init, b_init, act = nothing)
    weight = getweight(w_init, Array, state_dict, joinname(prefix, "weight"))
    if isnothing(b_init)
        bias = nothing
    else
        bias = getweight(b_init, Array, state_dict, joinname(prefix, "bias"))
    end
    return Layers.Dense(act, weight, bias)
end
