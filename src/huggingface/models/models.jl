import InteractiveUtils
using LinearAlgebra
using ValSplit
using StructWalk
using Functors
using DataStructures: OrderedDict
using Pickle

using ..Layers: @fluxshow, @fluxlayershow

include("./load.jl")

"""
  `get_model_type(model_type)`

See the list of supported model type of given model. For example, use `get_mdoel_type(:gpt2)` to see all model/task
 that `gpt2` support. The `keys` of the returned `NamedTuple` are all possible task which can be used in
 [`load_model`](@ref) or [`@hgf_str`](@ref).

# Example

```julia-repl
julia> HuggingFace.get_model_type(:gpt2)
(model = Transformers.HuggingFace.HGFGPT2Model, lmheadmodel = Transformers.HuggingFace.HGFGPT2LMHeadModel)

```
"""
get_model_type

_symbol(v::Val) = v
_symbol(v) = Symbol(v)

function _get_model_type(model_type::Symbol)
    types = Tuple(InteractiveUtils.subtypes(HGFPreTrained{model_type}))
    isempty(types) &&
        error("Unknown model type: $model_type")
    return NamedTuple{getmodeltask.(types)}(types)
end

@valsplit get_model_type(Val(model_type::Symbol)) = _get_model_type(model_type)
function get_model_type(model_type, task::Union{Symbol, String})
    task = Symbol(lowercase(String(task)))
    tasks = get_model_type(_symbol(model_type))
    if haskey(tasks, task)
        getfield(tasks, task)
    else
        error("Model $model_type doesn't support this kind of task: $task")
    end
end

load_model(model_name; kws...) = load_model(model_name, :model; kws...)
function load_model(model_name, task; config = nothing, kws...)
    if isnothing(config)
        config = load_config(model_name; kws...)
    end
    model_type = getconfigname(config)
    return load_model(model_type, model_name, task; config, kws...)
end
function load_model(model_name::AbstractString, task::Symbol, state_dict; trainmode = false, config = nothing, kws...)
    if isnothing(config)
        config = load_config(model_name; kws...)
    end
    model_type = getconfigname(config)
    return load_model(model_type, model_name, task, state_dict; trainmode, config, kws...)
end
function load_model(model_type, model_name::AbstractString, task; trainmode = false, config = nothing, mmap = true, lazy = mmap, kws...)
    if isnothing(config)
        config = load_config(model_name; kws...)
    end
    state_dict = load_state_dict(model_name; lazy, mmap, kws...)
    return load_model(model_type, model_name, task, state_dict; trainmode, config, kws...)
end
function load_model(model_type, model_name::AbstractString, task, state_dict;
                    trainmode = false, config = nothing, kws...)
    if isnothing(config)
        config = load_config(model_name; kws...)
    end
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
  `save_model(model_name, model; path = pwd(), weight_name = PYTORCH_WEIGHTS_NAME, force = false)`

save the `model` state_dict at `<path>/<model_name>/<weight_name>`. This would error out if the file already exists
 but `force` not set.
"""
function save_model(model_name, model; path = pwd(), weight_name = PYTORCH_WEIGHTS_NAME)
    model_path = joinpath(path, model_name)
    mkpath(model_path)
    model_file = joinpath(model_path, weight_name)
    if isfile(model_file)
        if force
            @warn "$model_file forcely updated."
        else
            error("$model_file already exists. set `force = true` for force update.")
        end
    end
    state = get_state_dict(model)
    Pickle.Torch.THsave(model_file, state)
    return model_file
end

function is_seq2seq(model)
    has_seq2seq = Ref(false)
    StructWalk.scan(Layers.LayerStyle, model) do x
        x isa Layers.Seq2Seq && (has_seq2seq[] = true)
    end
    return has_seq2seq[]
end

# api doc

"""
    load_model([model_type::Symbol,] model_name, task = :model [, state_dict];
               trainmode = false, config = nothing, local_files_only = false, cache = true)

Load the model of `model_name` for `task`. This function would load the `state_dict` of `model_name` and build a new
 model according to `config`, `task`, and the `state_dict`. `local_files_only` and `cache` kwargs would be pass directly
 to both [`load_state_dict`](@ref) and [`load_config`](@ref) if not provided. This function would require the
 configuration file has a field about the `model_type`, if not, use `load_model(model_type, model_name, task; kwargs...)`
 with `model_type` manually provided. `trainmode = false` would disable all dropouts. The `state_dict` can be directly
 provided, this is used when you want to create a new model with the `state_dict` in hand. Use [`get_model_type`](@ref)
 to see what `task` is available.

See also: [`get_model_type`](@ref), [`load_state_dict`](@ref), [`load_config`](@ref), [`HGFConfig`](@ref)
"""
load_model

"""
    get_state_dict(model)

Get the state_dict of the model.
"""
get_state_dict

"""
    load_model(::Type{T}, config, state_dict = OrderedDict())

Create a new model of `T` according to `config` and `state_dict`. missing parameter would initialized according
 to `config`. Set the `JULIA_DEBUG=Transformers` environment variable to see what parameters are missing.
"""
load_model(::Type, cfg)
