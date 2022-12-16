using Flux
using Functors
using DataStructures
using Pickle.Torch
using Pickle.Torch: StridedView

using ValSplit

using LinearAlgebra

"""
  `get_state_dict(layer)`

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

include("./utils.jl")
include("./base.jl")

"""
  `get_model_type(::Val{model})`

See the list of supported model type of given model.
For example, use `get_mdoel_type(Val(:bert))` to
see all model/task that `bert` support.
"""
get_model_type

@valsplit get_model_type(Val(model_type::Symbol)) = error("Unknown model type: $model_type")
function get_model_type(model_type, task::Symbol)
    task = Symbol(lowercase(String(task)))
    tasks = get_model_type(model_type)
    if haskey(tasks, task)
        getfield(tasks, task)
    else
        error("Model $model_type doesn't support this kind of task: $task")
    end
end

include("./load.jl")

load_model(model_name; kws...) = load_model(model_name, :model; kws...)
function load_model(model_name, task; config = nothing, kws...)
    if isnothing(config)
        config = load_config(model_name; kws...)
    end
    model_type = getconfigname(config)
    return load_model(model_type, model_name, task; config, kws...)
end
function load_model(model_type, model_name::AbstractString, task; trainmode = false, config = nothing, kws...)
    if isnothing(config)
        config = load_config(model_name; kws...)
    end
    state_dict = load_state_dict(model_name; kws...)
    T = get_model_type(model_type, task)
    basekey = String(basemodelkey(T))
    if isbasemodel(T)
        prefix = haskeystartswith(state_dict, basekey) ? basekey : ""
    else
        prefix = ""
        if !haskeystartswith(state_dict, basekey)
            new_state_dict = OrderedDict{Any, Any}()
            for (key, val) in state_dict
                new_state_dict[joinname(basekey, key)] = val
            end
            state_dict = new_state_dict
        end
    end
    model = load_model(T, config, state_dict, prefix)
    trainmode || (model = Layers.testmode(model))
    return model
end

"""
  `save_model(model_name, model; path = pwd(), weight_name = PYTORCH_WEIGHTS_NAME)`

save the `model` at `<path>/<model_name>/<weight_name>`.
"""
function save_model(model_name, model; path = pwd(), weight_name = PYTORCH_WEIGHTS_NAME)
  model_path = joinpath(path, model_name)
  !isdir(model_path) && error("$model_path is not a dir.")
  model_file = joinpath(model_path, weight_name)
  state = get_state_dict(model)
  Torch.THsave(model_file, state)
  return model_file
end

is_seq2seq(_) = false
