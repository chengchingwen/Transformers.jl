using AbstractTrees
import AbstractTrees: children

const ACT2FN = (gelu = gelu, relu = relu, swish = swish, gelu_new = gelu, mish = mish)

abstract type THModule end
abstract type HGFPreTrainedModel <: THModule end

struct FakeTHLayerNorm{T<:AbstractArray} <: THModule
  eps::Float32
  weight::T
  bias::T
end

Functors.functor(::Type{<:FakeTHLayerNorm}, layernorm) = (weight = layernorm.weight, bias = layernorm.bias), y -> FakeTHEmbedding(layernorm.eps, y...)

function load_state(layer::FakeTHLayerNorm, state)
  for k in keys(state)
    if k == :gamma
      nk = :weight
    elseif k == :beta
      nk = :bias
    else
      nk = k
    end
    load_state(getfield(layer, nk), getfield(state, k))
  end
end

struct FakeTHLinear{W<:AbstractArray, B<:Union{Nothing, AbstractArray}} <: THModule
  weight::W
  bias::B
end

FakeTHLinear(w) = FakeTHLinear(w, nothing)

_has_bias(::FakeTHLinear{W, Nothing}) where {W<:AbstractArray}= false
_has_bias(::FakeTHLinear) = true

function Functors.functor(::Type{<:FakeTHLinear}, linear)
  (_has_bias(linear) ? (weight = linear.weight, bias = linear.bias) : (weight = linear.weight,)),
  y -> FakeTHLinear(y...)
end

struct FakeTHEmbedding{T<:AbstractArray} <: THModule
  pad_idx::Union{Nothing, Int}
  weight::T
end

Functors.functor(::Type{<:FakeTHEmbedding}, embedding) = (weight = embedding.weight,), y -> FakeTHEmbedding(embedding.pad_idx, y...)

function load_state(layer::FakeTHEmbedding, state)
  load_state(layer.weight, state.weight')
end

function get_state_dict(state, prefix, embedding::FakeTHEmbedding)
  param = Functors.functor(embedding)[1]
  k = :weight
  cprefix = isnothing(prefix) ? String(k) : join((prefix, k), '.')
  state[cprefix] = param.weight'
end

struct FakeTHModuleList <: THModule
  _modules::Vector
end

Functors.functor(::Type{<:FakeTHModuleList}, modulelist) = modulelist._modules, y -> FakeTHModuleList(y)

Base.iterate(modulelist::FakeTHModuleList) = iterate(modulelist._modules)
Base.iterate(modulelist::FakeTHModuleList, i...) = iterate(modulelist._modules, i...)

function get_state_dict(state, prefix, modulelist::FakeTHModuleList)
  param = Functors.functor(modulelist)[1]
  for (i, layer) in enumerate(modulelist)
    cprefix = join((prefix, i-1), '.')
    get_state_dict(state, cprefix, layer)
  end
end

function load_state(layer::FakeTHModuleList, state)
  for (layerᵢ, stateᵢ) in zip(layer, state)
    load_state(layerᵢ, stateᵢ)
  end
end

function Base.show(io::IO, x::M; depth=10) where {M<:THModule}
  print_tree(_printnode, io, x, depth)
  io
end

children(m::THModule) = Tuple(pairs(first(Functors.functor(m))))
function children(::THModule, x::Pair{Symbol})
  v = x[2]
  v isa AbstractArray ? () : children(v)
end

_printnode(io, x) = summary(io, x)
function _printnode(io::IO, p::Pair)
  print(io, p[1])
  print(io, " => ")
  summary(io, p[2])
end


