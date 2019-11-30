using Flux: OneHotVector, onehot

import Base

struct OneHotArray{N, A<: AbstractArray{OneHotVector}} <: AbstractArray{Bool, N}
    dims::Int
    data::A
    OneHotArray(dims, data::A) where A = new{ndims(A)+1, A}(dims, data)
end

Base.size(xs::OneHotArray) = (Int64(xs.dims), size(xs.data)...)

Base.getindex(o::OneHotArray, i::Integer) = o[CartesianIndices(o)[i]]
Base.getindex(o::OneHotArray, i::Integer, j::Integer) = o.data[j][i]
Base.getindex(o::OneHotArray, i::Integer, j::Integer, I::Vararg{Int, N}) where N = o.data[j, I...][i]
Base.getindex(o::OneHotArray, ::Colon, i::Integer) = o.data[i]
Base.getindex(o::OneHotArray, ::Colon, i::AbstractArray) = OneHotArray(o.dims, o.data[i])


OneHotArray(xs::A) where A = OneHotArray{ndims(A)+1, A}(Int(xs[1].of), xs)
OneHotArray(nums::Int, xs::AbstractArray{Int}) = OneHotArray(nums, map(i->onehot(i, 1:nums), xs))

onehotarray(num::Int, xs::AbstractArray{Int}) = OneHotArray(num, indices2onehot(num, xs))

import Adapt: adapt, adapt_structure

adapt_structure(T, xs::OneHotArray) = OneHotArray(xs.dims, adapt(T, xs.data))

tofloat(::Type{F}, o::OneHotArray{N, <:AbstractArray}) where {F<:AbstractFloat,N} = Array{F}(o)

Base.convert(::Type{OneHotVector}, x::Int) = OneHotVector(x & 0xffffffff, x >> 32)
Base.convert(::Type{Int}, x::OneHotVector) = Int(x.ix)

Base.hcat(x::OneHotArray, xs::OneHotArray...) = begin
    !all(isequal(x.dims), map(o->o.dims, xs)) && throw(DimensionMismatch("OneHot dimension are not all the same"))
    OneHotArray(x.dims, vcat(x.data, map(o->o.data, xs)...))
end

Base.cat(x::OneHotArray, xs::OneHotArray...; dims::Int) = begin
    !all(isequal(x.dims), map(o->o.dims, xs)) && throw(DimensionMismatch("OneHot dimension are not all the same"))
    dims < 2 && error("OneHotArray concatenation at dimension $dims is not allow")
    OneHotArray(x.dims, cat(x.data, map(o->o.data, xs)...; dims=dims-1))
end


"turn one hot encoding to indices"
onehot2indices(xs::OneHotArray) = onehot2indices(xs.data)

#cpu onehot to indices
onehot2indices(x::AbstractArray{OneHotVector}) = map(i->Int(i.ix), x)

#cpu indices to onehot
indices2onehot(nums::Int, xs::AbstractArray{Int}) = map(i->onehot(i, 1:nums), xs)

_labelindex(num::Int, x::AbstractArray) = x .+ (num << 32)

using ZygoteRules: @adjoint, AContext, pullback
import ZygoteRules: _pullback

import Base: *
*(A::AbstractMatrix{T}, b::OneHotArray{N}) where {T,N} = gather(A, b)

_pullback(cx::AContext, ::typeof(*), A::AbstractMatrix{T}, b::OneHotArray{N}) where {T, N} = _pullback(cx, gather, A, b)
