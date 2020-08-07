using Flux
using Functors
using DataStructures
using Pickle.Torch: StridedView

using LinearAlgebra

function get_state_dict(layer)
  state = OrderedDict{String, Any}()
  get_state_dict(state, nothing, layer)
  return state
end

function get_state_dict(state, prefix, layer)
  param = Functors.functor(layer)[1]
  ks = keys(param)
  for k in ks
    cprefix = isnothing(prefix) ? String(k) : join((prefix, k), '.')
    get_state_dict(state, cprefix, param[k])
  end
end

function get_state_dict(state, prefix, x::AbstractArray)
  state[prefix] = x
end

function load_state(layer, state)
  for k in keys(state)
    if hasfield(typeof(layer), k)
      load_state(getfield(layer, k),  getfield(state, k))
    end
  end
end

function load_state(weight::A1, state::A2) where {A1<:AbstractArray, A2<:AbstractArray}
    weight .= state
end

function load_state(weight::T, state::S) where {T<:Transpose, S<:StridedView}
  # weight is probably share weighted
  if weight != state
    weight .= state
  end
end

include("./utils.jl")
include("./base.jl")
include("./bert.jl")
