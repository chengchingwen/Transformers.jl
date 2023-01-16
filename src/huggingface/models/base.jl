using Statistics
using ..Transformers.Basic: gather

using Flux: @adjoint, pullback

using AbstractTrees
import AbstractTrees: children

"""
    quick_gelu(x)

Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs

Implementation in HuggingFace
https://github.com/huggingface/transformers/blob/9e56aff58a742b48fc8edea8d28d5b80330efbcc/src/transformers/activations.py#L69

"""
function quick_gelu(x)
  x * sigmoid_fast(1.702f0 * x)
end

const ACT2FN = (gelu = gelu, relu = relu, swish = swish, gelu_new = gelu, mish = mish, quick_gelu = quick_gelu, selu = selu)

abstract type THModule end
abstract type HGFPreTrainedModel <: THModule end

# torch.nn.LayerNorm

struct FakeTHLayerNorm{T<:AbstractArray} <: THModule
  eps::Float32
  weight::T
  bias::T
end

Functors.functor(::Type{<:FakeTHLayerNorm}, layernorm) = (weight = layernorm.weight, bias = layernorm.bias), y -> FakeTHLayerNorm(layernorm.eps, y...)

(ln::FakeTHLayerNorm)(x) = ln.weight .* Flux.normalise(x, dims=1, ϵ=ln.eps) .+ ln.bias

function load_state!(layer::FakeTHLayerNorm, state)
  for k in keys(state)
    if k == :gamma
      nk = :weight
    elseif k == :beta
      nk = :bias
    else
      nk = k
    end
    load_state!(getfield(layer, nk), getfield(state, k))
  end
end

# torch.nn.Linear

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

# torch.nn.Embedding

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

function load_state!(layer::FakeTHEmbedding, state)
  load_state!(layer.weight, state.weight')
end

function get_state_dict(state, prefix, embedding::FakeTHEmbedding)
  param = Functors.functor(embedding)[1]
  k = :weight
  cprefix = isnothing(prefix) ? String(k) : join((prefix, k), '.')
  state[cprefix] = param.weight'
end

# torch.nn.ModuleList

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

function load_state!(layer::FakeTHModuleList, state)
  for (layerᵢ, stateᵢ) in zip(layer, state)
    load_state!(layerᵢ, stateᵢ)
  end
end

# displaying layer in tree structure

using Pkg
pkgversion(m::Module) = _pkgversion(joinpath(dirname(string(first(methods(Base.moduleroot(m).eval)).file)), "..", "Project.toml"))
function _pkgversion(project)
    if project !== nothing && isfile(project)
        toml = Pkg.TOML.parsefile(project)
        haskey(toml, "version") && return Base.VersionNumber(toml["version"])
    end
    return nothing
end

if pkgversion(AbstractTrees) < v"0.4"
    function Base.show(io::IO, x::M; depth=10) where {M<:THModule}
        print_tree(_printnode, io, x, depth)
        io
    end
    children(m::THModule) = Tuple(pairs(first(Functors.functor(m))))
    function children(::THModule, x::Pair{Symbol})
        v = x[2]
        v isa AbstractArray ? () : children(v)
    end
    children(::THModule, x::Pair{Int}) = children(x[2])
else
    function Base.show(io::IO, x::M; depth=10) where {M<:THModule}
        print_tree(_printnode, io, x; maxdepth = depth)
        io
    end
    AbstractTrees.nodevalue(m::THModule) = Base.nameof(typeof(m))
    children(m::THModule) = Tuple(pairs(first(Functors.functor(m))))
    function children(x::Pair{Symbol})
        v = x[2]
        v isa AbstractArray ? () : children(v)
    end
    children(x::Pair{Int}) = children(x[2])
end

_printnode(io, x) = print(io, Base.nameof(typeof(x)))
function _printnode(io::IO, p::Pair)
  print(io, p[1])
  print(io, " => ")
  v = p[2]
  if v isa AbstractArray
    summary(io, p[2])
  else
    print(io, Base.nameof(typeof(v)))
  end
end

# transformers.modeling_utils.Conv1D

struct FakeHGFConv1D{W<:AbstractArray, B<:AbstractArray} <: THModule
  weight::W
  bias::B
end

@functor FakeHGFConv1D

(c::FakeHGFConv1D)(x::AbstractMatrix) = c.weight' * x .+ c.bias

function (c::FakeHGFConv1D)(x::AbstractArray)
  old_size = size(x)
  new_size = Base.setindex(old_size, size(c.weight, 2), 1)

  new_x = reshape(x, old_size[1], :)
  y = c(new_x)
  return reshape(y, new_size)
end

# transformers.modeling_utils.SequenceSummary

abstract type AbstractSummaryType end
struct LastSummary <: AbstractSummaryType end
struct FirstSummary <: AbstractSummaryType end
struct MeanSummary <: AbstractSummaryType end
struct CLSindexSummary <: AbstractSummaryType end

struct FakeHGFSequenceSummary{T<:AbstractSummaryType, S, F} <: THModule
  summary::S
  activation::F

  FakeHGFSequenceSummary{T}(s, a) where {T} = new{T, typeof(s), typeof(a)}(s, a)
end

SummaryType(s::Symbol) = (LastSummary, FirstSummary, MeanSummary, CLSindexSummary)[
  findfirst(isequal(s), (:last, :first, :mean, :cls_index))
]

summarytype(s::FakeHGFSequenceSummary{T}) where {T<:AbstractSummaryType} = T

Functors.functor(::Type{<:FakeHGFSequenceSummary}, ss) = (summary = ss.summary, activation = ss.activation), y -> FakeHGFSequenceSummary{summarytype(ss)}(y...)

summary_input(s::FakeHGFSequenceSummary{LastSummary}, x) = x[:, end, :, :]
summary_input(s::FakeHGFSequenceSummary{FirstSummary}, x) = x[:, 1, :, :]
summary_input(s::FakeHGFSequenceSummary{MeanSummary}, x) = mean(x, dims=2)

summary_input(s::FakeHGFSequenceSummary, x, _) = summary_input(s, x)
summary_input(s::FakeHGFSequenceSummary{CLSindexSummary}, x, ::Nothing) = x[:, end, :, :]
summary_input(s::FakeHGFSequenceSummary{CLSindexSummary}, x, cls_index) = gather(x, cls_index)

function (s::FakeHGFSequenceSummary)(hidden_states, cls_index)
  output = summary_input(s, hidden_states, cls_index)
  output = s.summary(output)
  output = s.activation(output)
  return output
end
