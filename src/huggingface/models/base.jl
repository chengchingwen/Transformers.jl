using ..Transformers.Basic: gather

using ZygoteRules: @adjoint, pullback

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

Functors.functor(::Type{<:FakeTHLayerNorm}, layernorm) = (weight = layernorm.weight, bias = layernorm.bias), y -> FakeTHLayerNorm(layernorm.eps, y...)

(ln::FakeTHLayerNorm)(x) = ln.weight .* Flux.normalise(x, dims=1, ϵ=ln.eps) .+ ln.bias

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

(l::FakeTHLinear)(x::AbstractMatrix) = _has_bias(l) ? l.weight * x .+ l.bias : l.weight * x
function (l::FakeTHLinear)(x::AbstractArray)
  old_size = size(x)
  new_size = Base.setindex(old_size, size(l.weight, 1), 1)

  new_x = reshape(x, old_size[1], :)
  y = l(new_x)
  return reshape(y, new_size)
end

struct FakeTHEmbedding{T<:AbstractArray} <: THModule
  pad_idx::Int
  weight::T
end

function FakeTHEmbedding(pad_idx, weight)
  @assert pad_idx <= size(weight, 2)
  if !iszero(pad_idx)
    @view(weight[:, pad_idx]) .= 0
  end

  FakeTHEmbedding(pad_idx, weight)
end

Functors.functor(::Type{<:FakeTHEmbedding}, embedding) = (weight = embedding.weight,), y -> FakeTHEmbedding(embedding.pad_idx, y...)

#TODO: refactor basic/embeds/gather.jl
_padded_gather(w, x, padded_idx) = gather(w, x)
@adjoint function _padded_gather(w, x, padded_idx)
  y, back = pullback(gather, w, x)
  return y, Δ -> begin
    Δ′, _ = back(Δ)
    if !iszero(padded_idx)
      @view(Δ′[:, padded_idx]) .= 0
    end
    return (Δ′, nothing, nothing)
  end
end

(e::FakeTHEmbedding)(x) = _padded_gather(e.weight, x, e.pad_idx)

function load_state(layer::FakeTHEmbedding, state)
  load_state(layer.weight, state.weight')
end

function get_state_dict(state, prefix, embedding::FakeTHEmbedding)
  param = Functors.functor(embedding)[1]
  k = :weight
  cprefix = isnothing(prefix) ? String(k) : join((prefix, k), '.')
  state[cprefix] = param.weight'
end

struct FakeTHModuleList{N, T<:Tuple} <: THModule
  _modules::T
  FakeTHModuleList(ms) = new{length(ms), typeof(ms)}(ms)
end

FakeTHModuleList(ms...) = FakeTHModuleList(ms)
FakeTHModuleList(ms::Vector) = FakeTHModuleList(Tuple(ms))

Functors.functor(::Type{<:FakeTHModuleList}, modulelist) = modulelist._modules, y -> FakeTHModuleList(y)

Base.iterate(modulelist::FakeTHModuleList) = iterate(modulelist._modules)
Base.iterate(modulelist::FakeTHModuleList, i...) = iterate(modulelist._modules, i...)
Base.length(::FakeTHModuleList{N}) where N = N
Base.getindex(modulelist::FakeTHModuleList, i) = modulelist._modules[i]

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

struct FakeHGFConv1D{W<:AbstractArray, B<:AbstractArray} <: THModule
  weight::W
  bias::B
end

@functor FakeHGFConv1D

(c::FakeHGFConv1D)(x::AbstractMatrix) = l.weight' * x .+ l.bias

function (c::FakeHGFConv1D)(x::AbstractArray)
  old_size = size(x)
  new_size = Base.setindex(old_size, size(l.weight, 2), 1)

  new_x = reshape(x, old_size[1], :)
  y = l(new_x)
  return reshape(y, new_size)
end
