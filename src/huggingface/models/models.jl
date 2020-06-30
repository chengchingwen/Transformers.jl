using Flux
using Functors
using Pickle.Torch: StridedView

using LinearAlgebra

const ACT2FN = (gelu = gelu, relu = relu, swish = swish, gelu_new = gelu, mish = mish)

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

function load_state(weight::A, state::S) where {A<:AbstractArray, S<:StridedView}
  # state is probably row major array
  weight .= PermutedDimsArray(state, ndims(state):-1:1)
end

function load_state(weight::T, state::S) where {T<:Transpose, S<:StridedView}
  # weight is probably share weighted
  if weight != state
    weight .= state
  end
end

struct FakeTHLayerNorm{T<:AbstractArray}
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
    getfield(layer, nk) .= getfield(state, k)
  end
end

struct FakeTHLinear{W<:AbstractArray, B<:Union{Nothing, AbstractArray}}
  weight::W
  bias::B
end

FakeTHLinear(w) = FakeTHLinear(w, nothing)

_has_bias(::FakeTHLinear{W, Nothing}) where W = false
_has_bias(::FakeTHLinear) = true

function Functors.functor(::Type{<:FakeTHLinear}, linear)
  (_has_bias(linear) ? (weight = linear.weight, bias = linear.bias) : (weight = linear.weight,)),
  y -> FakeTHLinear(y...)
end

struct FakeTHEmbedding{T<:AbstractArray}
  pad_idx::Union{Nothing, Int}
  weight::T
end

Functors.functor(::Type{<:FakeTHEmbedding}, embedding) = (weight = embedding.weight,), y -> FakeTHEmbedding(embedding.pad_idx, y...)

struct FakeTHModuleList
  _modules::Vector
end

Functors.functor(::Type{<:FakeTHModuleList}, modulelist) = modulelist._modules, y -> FakeTHModuleList(y)

Base.iterate(modulelist::FakeTHModuleList) = iterate(modulelist._modules)
Base.iterate(modulelist::FakeTHModuleList, i...) = iterate(modulelist._modules, i...)

function load_state(layer::FakeTHModuleList, state)
  for (layerᵢ, stateᵢ) in zip(layer, state)
    load_state(layerᵢ, stateᵢ)
  end
end


include("./model_bert.jl")
