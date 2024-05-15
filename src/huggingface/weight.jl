using DataStructures: OrderedDict
using JSON3
using Pickle
using SafeTensors

"""
  `load_state_dict(model_name; local_files_only = false, force_format = :auto, cache = true)`

Load the `state_dict` from the given `model_name` from huggingface hub. By default, this function would check if
 `model_name` exists on huggingface hub, download the model file (and cache it if `cache` is set), and then load
 and return the `state_dict`. If `local_files_only = false`, it would check whether the model file is up-to-date and
 update if not (and thus require network access every time it is called). By setting `local_files_only = true`, it
 would try to find the files from the cache directly and error out if not found. For managing the caches, see the
 `HuggingFaceApi.jl` package. If `force_format` is `:auto` it will automatically selects the format from which the
 weights will be loaded. If `force_format` is `:pickle` or `:safetensor`, it will prefer relevant file.
"""
function load_state_dict(model_name; lazy = false, mmap = true, possible_files = nothing, force_format = :auto, kw...)
    possible_files = ensure_possible_files(possible_files, model_name; kw...)
    weight_format = force_format == :auto ? detect_weight_format(model_name; possible_files, kw...) : force_format
    status = WeightStatus{weight_format}(model_name; possible_files, kw...)
    if status isa HasWeightIn
        return load_state_dict_from(status; lazy, mmap, kw...)
    else
        error("The repository does not contain the weights stored in $(weight_format) format")
    end
end

function detect_weight_format(model_name; possible_files = nothing, kws...)
    HasWeightIn{:pickle}(model_name; possible_files, kws...) && return :pickle
    HasWeightIn{:safetensor}(model_name; possible_files, kws...) && return :safetensor
    error("The repository does not contain the weights stored in supported formats (pytorch pickle or safetensors)")
end

abstract type WeightStatus{format} end
abstract type HasWeightIn{format} <: WeightStatus{format} end
struct HasIndexMap{format} <: HasWeightIn{format}
    indexmap::Dict{String, Union{Nothing, Set{String}}}
end
struct HasSingleFile{format} <: HasWeightIn{format}
    file::String
end
struct NoWeightIn{format} <: WeightStatus{format} end

indexmapname(::Type{WeightStatus{:pickle}}) = PYTORCH_WEIGHTS_INDEX_NAME
indexmapname(::Type{WeightStatus{:safetensor}}) = SAFETENSOR_WEIGHTS_INDEX_NAME
singlefilename(::Type{WeightStatus{:pickle}}) = PYTORCH_WEIGHTS_NAME
singlefilename(::Type{WeightStatus{:safetensor}}) = SAFETENSOR_WEIGHTS_NAME
filelist(S::Type{WeightStatus{format}}) where format = (indexmapname(S), singlefilename(S))

function HasWeightIn{format}(model_name; possible_files = nothing, kw...) where format
    possible_files = ensure_possible_files(possible_files, model_name; kw...)
    return any(in(possible_files), filelist(WeightStatus{format})) ? true : false
end
function WeightStatus{format}(model_name; possible_files = nothing, kw...) where format
    possible_files = ensure_possible_files(possible_files, model_name; kw...)
    if indexmapname(WeightStatus{format}) in possible_files
        weightmap = JSON3.read(read(hgf_file(model_name, indexmapname(WeightStatus{format}); kw...))).weight_map
        indexmap = Dict{String, Set{String}}()
        for (weight, filename) in weightmap
            keyset = get!(()->Set{String}(), indexmap, hgf_file(model_name, filename; kw...))
            push!(keyset, weight)
        end
        return HasIndexMap{format}(indexmap)
    elseif singlefilename(WeightStatus{format}) in possible_files
        return HasSingleFile{format}(hgf_file(model_name, singlefilename(WeightStatus{format}); kw...))
    else
        return NoWeightIn{format}()
    end
end

function _state_dict_add!(state_dict, key, val; file = nothing)
    if haskey(state_dict, key)
        @warn """weight "$key" is overwritten by weight $(isnothing(file) ? "" : "from file \"$file\" ")with the same name."""
    end
    state_dict[key] = val
end
function _debug_key_misalign(file, notfound, unexpected)
    if !(isempty(unexpected) && isempty(notfound))
        @debug(
            "$file contains/missing some weights",
            var"expected to contain but not found" = notfound,
            var"unexpected but appear in file" = unexpected,
        )
    end
end

load_state_dict_from(status::HasSingleFile; lazy = false, mmap = true, keyset = nothing, kw...) =
    load_state_dict_from!(status, OrderedDict{Any, Any}(); lazy, mmap, kw...)
function load_state_dict_from!(status::HasSingleFile{:pickle}, state_dict; lazy = false, mmap = true, keyset = nothing, kw...)
    file = status.file
    loaded_state_dict = Pickle.Torch.THload(file; lazy, mmap)
    for (key, val) in loaded_state_dict
        _state_dict_add!(state_dict, key, val; file)
    end
    if !isnothing(keyset)
        loaded_keys = keys(loaded_state_dict)
        unexpected = setdiff(loaded_keys, keyset)
        notfound = setdiff(keyset, loaded_keys)
        _debug_key_misalign(file, notfound, unexpected)
    end
    return state_dict
end
function load_state_dict_from!(status::HasSingleFile{:safetensor}, state_dict; lazy = false, mmap = true, keyset = nothing, kw...)
    file = status.file
    safetensor = SafeTensors.deserialize(file; mmap)
    stored2shared = Dict{String, Set{String}}()
    # https://github.com/huggingface/safetensors/blob/b947b59079a6197d7930dfb535818ac4896113e8/bindings/python/py_src/safetensors/torch.py#L155-L165
    # huggingface seems to store shared tensor name in the metadata, so we tried to restore the information
    #  by checking if the metadata contain the tensor names.
    for (metakey, metaval) in safetensor.metadata # removed => kept
        if haskey(safetensor, metaval)
            sharednames = get!(()->Set{String}(), stored2shared, metaval)
            push!(sharednames, metakey)
        end
    end
    for (key, val) in safetensor
        val = lazy ? val : collect(val)
        _state_dict_add!(state_dict, key, val; file)
        if haskey(stored2shared, key)
            for sharedname in stored2shared[key]
                _state_dict_add!(state_dict, sharedname, val; file)
            end
        end
    end
    if !isnothing(keyset)
        loaded_keys = union(keys(safetensor), values(stored2shared)...)
        unexpected = setdiff(loaded_keys, keyset)
        notfound = setdiff(keyset, loaded_keys)
        _debug_key_misalign(file, notfound, unexpected)
    end
    return state_dict
end
load_state_dict_from(status::HasIndexMap; lazy = false, mmap = true, kw...) =
    load_state_dict_from!(status, OrderedDict{Any, Any}(); lazy, mmap, kw...)
function load_state_dict_from!(status::HasIndexMap{format}, state_dict; lazy = false, mmap = true, kw...) where format
    for (file, keyset) in status.files
        load_state_dict_from!(HasSingleFile{format}(file), state_dict; lazy, mmap, keyset, kw...)
    end
    return state_dict
end


"""
  `state_dict_to_namedtuple(state_dict)`

convert state_dict into nested `NamedTuple`.
"""
function state_dict_to_namedtuple(state_dict)
  ht = Pickle.HierarchicalTable()
  foreach(((k, v),)->setindex!(ht, v, k), pairs(state_dict))
  _ht2nt(ht)
end

_ht2nt(x::Some) = something(x)
_ht2nt(x::Pickle.HierarchicalTable) = _ht2nt(x.head)
function _ht2nt(x::Pickle.TableBlock)
  if iszero(length(x.entry))
    return ()
  else
    tks = Tuple(keys(x.entry))
    if all(Base.Fix1(all, isdigit), tks)
      inds = Vector(undef, length(tks))
      foreach(tks) do is
        i = parse(Int, is) + 1
        inds[i] = _ht2nt(x.entry[is])
      end
      return inds
    else
      cs = map(_ht2nt, values(x.entry))
      ns = map(Symbol, tks)
      return NamedTuple{ns}(cs)
    end
  end
end
