using Pickle
using Pickle: TableBlock, HierarchicalTable

"""
  `load_state(model_name)`

load the state from the given model name as NamedTuple.
"""
function load_state(model_name)
  state_dict = load_state_dict(model_name)
  state = state_dict_to_namedtuple(state_dict)
  return state
end

"""
  `load_state_dict(model_name)`

load the state_dict from the given model name.

See also: [`state_dict_to_namedtuple`](@ref)
"""
function load_state_dict(model_name)
  state_dict = Pickle.Torch.THload(get_registered_weight_path(model_name))
  return state_dict
end

"""
  `state_dict_to_namedtuple(state_dict)`

convert state_dict into NamedTuple.
"""
function state_dict_to_namedtuple(state_dict)
  ht = Pickle.HierarchicalTable()
  foreach(((k, v),)->setindex!(ht, v, k), pairs(state_dict))
  _ht2nt(ht)
end

_ht2nt(x::Some) = something(x)
_ht2nt(x::HierarchicalTable) = _ht2nt(x.head)
function _ht2nt(x::TableBlock)
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
