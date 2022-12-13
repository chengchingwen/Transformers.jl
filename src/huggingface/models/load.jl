import ..Layers

using Flux
using Functors
using DataStructures
using Pickle.Torch
using Pickle.Torch: StridedView

using ValSplit

using LinearAlgebra

joinname(prefix, name) = isempty(prefix) ? name : join((prefix, name), '.')
joinname(prefix, n1, n2...) = joinname(prefix, join((n1, n2...), '.'))

const ObjTyp{T} = Union{T, Type{<:T}}

getweight(init, ::Type, ::Symbol) = init()
getweight(init,  x, sym::Symbol) = getproperty(x, sym)

getweight(init, ::Type{<:Array}, state_dict, name) = _getweight(collect, init, state_dict, name)
getweight(init, ::Type{<:Array}, state_dict::OrderedDict{String}, name) = getweight(init, state_dict, name)
getweight(init, ::Type{<:Layers.Embed}, state_dict, name) = _getweight(collect ∘ adjoint, init, state_dict, name)
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

get_state_dict(_, m::Layers.Dense, state_dict, prefix) = get_state_dict(m, state_dict, prefix)
function get_state_dict(m::Layers.Dense, state_dict, prefix)
    state_dict[joinname(prefix, "weight")] = m.W
    !isnothing(m.b) && (state_dict[joinname(prefix, "bias")] = m.b)
    return state_dict
end

get_state_dict(p, m::Layers.DropoutLayer, state_dict, prefix) = get_state_dict(p, m.layer, state_dict, prefix)

# @valsplit load_model(Val(type::Symbol), cfg::AbstractDict, state_dict = OrderedDict{String, Any}(); prefix = "") =
#     error("")

# function load_model(type::Type, cfg::AbstractDict, state_dict = OrderedDict{String, Any}();
#                     prefix::String = "")
#     @nospecialize type
# end

# function load_model(parent::Type, type::Type, cfg::AbstractDict, state_dict = OrderedDict{String, Any}();
#                     prefix::String = "")
#     @nospecialize parent type
# end
