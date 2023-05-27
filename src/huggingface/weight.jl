using DataStructures: OrderedDict
using JSON3
using Pickle

"""
  `load_state_dict(model_name; local_files_only = false, cache = true)`

Load the `state_dict` from the given `model_name` from huggingface hub. By default, this function would check if
 `model_name` exists on huggingface hub, download the model file (and cache it if `cache` is set), and then load
 and return the `state_dict`. If `local_files_only = false`, it would check whether the model file is up-to-date and
 update if not (and thus require network access every time it is called). By setting `local_files_only = true`, it
 would try to find the files from the cache directly and error out if not found. For managing the caches, see the
 `HuggingFaceApi.jl` package.
"""
function load_state_dict(model_name; possible_files = nothing, kw...)
    possible_files = ensure_possible_files(possible_files, model_name; kw...)
    if PYTORCH_WEIGHTS_INDEX_NAME in possible_files
        weight_index = JSON3.read(read(hgf_model_weight_index(model_name; kw...)))
        full_state_dict = OrderedDict{Any, Any}()
        for weight_file in Set{String}(values(weight_index.weight_map))
            merge!(full_state_dict, Pickle.Torch.THload(hgf_file(model_name, weight_file; kw...)))
        end
        return full_state_dict
    else
        return Pickle.Torch.THload(hgf_model_weight(model_name; kw...))
    end
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
