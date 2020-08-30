using Flux
using Functors
using DataStructures
using Pickle.Torch: StridedView

using LinearAlgebra

"""
  get_state_dict(layer)

Collect model parameters into one `OrderedDict` which also 
known as `state_dict` in PyTorch.

model parameters are get from `Functors.functor`.
"""
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

"""
  load_state!(layer, state)

Load the model parameter from `state` into the `layer`. 
give warning if something appear in `state` but not `layer`.
"""
function load_state!(layer, state)
  for k in keys(state)
    if hasfield(typeof(layer), k)
      load_state!(getfield(layer, k),  getfield(state, k))
    else
      @warn "$(Base.typename(typeof(layer))) doesn't have field $k."
    end
  end
  pf = Functors.functor(layer) |> first |> keys
  rem = setdiff(pf, keys(state))
  if (!iszero ∘ length)(rem)
    @warn "Some fields of $(Base.typename(typeof(layer))) aren't initialized with loaded state: $(rem...)"
  end
end

function load_state!(weight::A1, state::A2) where {A1<:AbstractArray, A2<:AbstractArray}
    weight .= state
end

function load_state!(weight::T, state::S) where {T<:Transpose, S<:StridedView}
  # weight is probably share weighted
  if weight != state
    weight .= state
  end
end

include("./utils.jl")
include("./base.jl")
include("./bert.jl")
include("./gpt2.jl")
include("./roberta.jl")

"""
  load_model!(model::HGFPreTrainedModel, state)

Similar to [`load_state!`](@ref) but only work for 
huggingface pre-trained models, this is used for 
handling loading state of different level.
"""
function load_model!(model::HGFPreTrainedModel, state)
  basekey = basemodelkey(model)

  load_to_base = isbasemodel(model)
  load_from_base = !haskey(state, basekey)

  if load_to_base ⊻ load_from_base # not same level
    if load_to_base
      basestate = state[basekey]
      load_state!(model, basestate)
    else # load_from_base
      model_to_load = basemodel(model)
      load_state!(model_to_load, state)
    end
  else
    load_state!(model, state)
  end
end

"""
  load_model(model_type, model_name; config=load_config(model_name))

build model with given `model_type` and load state from 
`model_name`.
"""
function load_model(model_type, model_name; config=load_config(model_name))
  model = model_type(config)
  state = load_state(model_name)
  load_model!(model, state)
  return model
end
