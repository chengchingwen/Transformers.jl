using Base: @_noinline_meta, @_inline_meta
using Core: is_top_bit_set
using Core.Intrinsics: bitcast, trunc_int, sext_int, zext_int, sle_int, eq_int, and_int

# onehot erorr
struct OneHotEncodeError <: Exception
  K
  val
  OneHotEncodeError(@nospecialize(K), @nospecialize(val)) = (@_noinline_meta; new(K, val))
end

function Base.showerror(io::IO, e::OneHotEncodeError)
  print(io, "OneHotEncodeError: cannot encode ")
  print(io, e.val)
  print(io, " with OneHot{")
  print(io, e.K)
  print(io, '}')
end

throw_onehotencode_error(K, val) = (@_noinline_meta; throw(OneHotEncodeError(K, val)))

# onehot encode
primitive type OneHot{K} <: AbstractVector{Bool} 32 end

OneHot(k) = OneHot{UInt32(k)}
OneHot{K}(x) where K = convert(OneHot(K), x)
OneHot(k, x) = OneHot{k}(x)

onehotsize(::OneHot{K}) where K = Int(K)

# array interface

Base.size(o::OneHot) = (onehotsize(o),)
function Base.getindex(o::OneHot{K}, i::I) where {K, I<:Integer}
  @boundscheck checkbounds(o, i)
  return convert(I, o) == i
end

# printing

function Base.showarg(io::IO, x::OneHot{K}, toplevel) where K
  toplevel || print(io, "::")
  join(io, ["OneHot{", Int(K), '}'])
end

# convert

Base.UInt32(o::OneHot) = bitcast(UInt32, o)
Base.UInt64(o::OneHot) = zext_int(UInt64, o)
Base.Int32(o::OneHot) = bitcast(Int32, o)
Base.Int64(o::OneHot) = zext_int(Int64, o)

Base.convert(::Type{Any}, o::OneHot) = o
Base.convert(::Type{OneHot{K}}, o::OneHot{K}) where {K} = o
Base.convert(::Type{UInt32},  o::OneHot) = UInt32(o)
Base.convert(::Type{To}, o::OneHot) where {To} = convert(To, convert(UInt32, o))

Base.convert(ot::Type{OneHot{K}}, x::Core.BuiltinInts) where {K} = toOneHot(ot, x)

# zero

Base.zero(o::O) where {O<:OneHot} = toOneHot(O, 0x00000000)
Base.iszero(o::O) where {O<:OneHot} = iszero(convert(UInt32, o))

# bit-op

function check_onehot_top_bit(::Type{OneHot{K}}, x) where {K}
  @_inline_meta
  is_top_bit_set(x) && throw_onehotencode_error(K, x)
  x
end

function check_onehot_encode(ot::Type{OneHot{K}}, x) where {K}
  @_inline_meta
  sle_int(x, K) || throw_onehotencode_error(K, x)
  bitcast(ot, x)
end

function checked_onehot_trunc_sint(ot::Type{OneHot{K}}, x::From) where {K, From}
  @_inline_meta
  y = trunc_int(UInt32, x)
  back = sext_int(From, y)
  eq_int(x, back) || throw_onehotencode_error(K, x)
  check_onehot_encode(ot, y)
end

function checked_onehot_trunc_uint(ot::Type{OneHot{K}}, x::From) where {K, From}
  @_inline_meta
  y = trunc_int(UInt32, x)
  back = zext_int(From, y)
  eq_int(x, back) || throw_onehotencode_error(K, x)
  check_onehot_encode(ot, y)
end

toOneHot(ot::Type{OneHot{K}}, x::Int8) where {K} = check_onehot_encode(ot, sext_int(UInt32, check_onehot_top_bit(ot, x)))
toOneHot(ot::Type{OneHot{K}}, x::Int16) where {K} = check_onehot_encode(ot, sext_int(UInt32, check_onehot_top_bit(ot, x)))
toOneHot(ot::Type{OneHot{K}}, x::Int32) where {K} = check_onehot_encode(ot, bitcast(UInt32, check_onehot_top_bit(ot, x)))
toOneHot(ot::Type{OneHot{K}}, x::Int64) where {K} = checked_onehot_trunc_sint(ot, x)
toOneHot(ot::Type{OneHot{K}}, x::Int128) where {K} = checked_onehot_trunc_sint(ot, x)
toOneHot(ot::Type{OneHot{K}}, x::UInt8) where {K} = check_onehot_encode(ot, zext_int(UInt32, x))
toOneHot(ot::Type{OneHot{K}}, x::UInt16) where {K} = check_onehot_encode(ot, zext_int(UInt32, x))
toOneHot(ot::Type{OneHot{K}}, x::UInt32) where {K} = check_onehot_encode(ot, x)
toOneHot(ot::Type{OneHot{K}}, x::UInt64) where {K} = checked_onehot_trunc_uint(ot, x)
toOneHot(ot::Type{OneHot{K}}, x::UInt128) where {K} = checked_onehot_trunc_uint(ot, x)
toOneHot(ot::Type{OneHot{K}}, x::Bool) where {K} = and_int(zext_int(ot, x), toOneHot(ot, 0x1))

# onehot array
struct OneHotArray{K, N, var"N+1", A<:AbstractArray{OneHot{K}, N}} <: AbstractArray{Bool, var"N+1"}
  onehots::A
end

OneHotArray(onehots::A) where {K, A<:AbstractArray{OneHot{K}}} = OneHotArray{K, ndims(onehots), ndims(onehots)+1, A}(onehots)
OneHotArray{K}(indices::A) where {K, A<:AbstractArray{<:Integer}} = OneHotArray(K, indices)
OneHotArray(k, xs) = OneHotArray(OneHot{k}.(xs))

onehotsize(::OneHotArray{K}) where K = Int(K)

# array interface
Base.size(oa::OneHotArray{K}) where K = (onehotsize(oa), size(oa.onehots)...)

function Base.getindex(oa::OneHotArray{K, N}, i, is::Vararg{Int, N}) where {K, N}
  @boundscheck checkbounds(oa, i, is...)
  oa.onehots[is...][i]
end

function Base.getindex(oa::OneHotArray{K, N}, i::Colon, is::Vararg{Int, N}) where {K, N}
  @boundscheck checkbounds(oa, i, is...)
  oa.onehots[is...]
end

function Base.getindex(oa::OneHotArray{K}, i::Colon, is...) where {K}
  @boundscheck checkbounds(oa, i, is...)
  OneHotArray(oa.onehots[is...])
end

Base.similar(o::OneHotArray, ::Type{T}, dims::Dims{N}) where {T, N} = similar(o.onehots, T, dims)

# printing

function Base.summary(io::IO, oa::OneHotArray)
  join(io, size(oa), 'x')
  join(io, [" OneHotArray{", onehotsize(oa), ", ", ndims(oa), ", "])
  Base.showarg(io, oa.onehots, true)
  print(io, "}")
end

# cat

function Base.cat(xss::OneHotArray{K}...; dims) where K
  isone(::Val{V}) where V = isone(V)
  isone(v) = Base.isone(v)
  if isone(dims)
    @warn "concat OneHotArray{$K} along dimension 1."
    Base._cat(Val(1), xss...)
  else
    predecessor(::Val{V}) where V = Val(V-1)
    predecessor(v) = v - 1
    sdims = predecessor(dims)
    xidss = map(xs->xs.onehots, xss)
    ret = cat(xidss...; dims=sdims)
    OneHotArray(ret)
  end
end

# old utility

onehot2indices(xs::OneHotArray) = convert(AbstractArray{Int}, xs.onehots)

indices2onehot(nums::Int, xs::AbstractArray{Int}) = OneHotArray(nums, xs)

tofloat(::Type{F}, o::OneHotArray{N, <:AbstractArray}) where {F<:AbstractFloat,N} = Array{F}(o)

# gpu
import Adapt: adapt, adapt_structure
adapt_structure(T, oa::OneHotArray) = OneHotArray(adapt(T, oa.onehots))

# AD
using Flux
Flux.@nograd OneHotArray

using ZygoteRules: @adjoint, AContext, pullback
import ZygoteRules: _pullback

import Base: *
*(A::AbstractMatrix{T}, b::OneHotArray{N}) where {T,N} = gather(A, b)

_pullback(cx::AContext, ::typeof(*), A::AbstractMatrix{T}, b::OneHotArray{N}) where {T, N} = _pullback(cx, gather, A, b)
