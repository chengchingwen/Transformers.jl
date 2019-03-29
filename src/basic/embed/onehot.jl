using CuArrays, CUDAnative
using Flux: OneHotVector, onehot

import Base

struct OneHotArray{N, A<: AbstractArray{OneHotVector}} <: AbstractArray{Bool, N}
    dims::Int
    data::A
    OneHotArray(dims, data::A) where A = new{ndims(A)+1, A}(dims, data)
end

Base.size(xs::OneHotArray) = (Int64(xs.dims), size(xs.data)...)

Base.getindex(o::OneHotArray, i::Integer, j::Integer) = o.data[j][i]
Base.getindex(o::OneHotArray, ::Colon, i::Integer) = o.data[i]
Base.getindex(o::OneHotArray, ::Colon, i::AbstractArray) = OneHotArray(o.dims, o.data[i])
Base.getindex(o::OneHotArray, i::Integer, j::Integer, k...) = o.data[j, k...][i]

OneHotArray(xs::A) where A = OneHotArray{ndims(A)+1, A}(Int(xs[1].of), xs)
OneHotArray(nums::Int, xs::AbstractArray{Int}) = OneHotArray(nums, map(i->onehot(i, 1:nums), xs))

onehotarray(num::Int, xs::AbstractArray{Int}) = OneHotArray(num, indices2onehot(num, xs))

import Adapt: adapt, adapt_structure

adapt_structure(T, xs::OneHotArray) = OneHotArray(xs.dims, adapt(T, xs.data))

#onehotarray(xs, labels)
Base.convert(::Type{OneHotVector}, x::Int) = OneHotVector(x & 0xffffffff, x >> 32)
Base.convert(::Type{Int}, x::OneHotVector) = Int(x.ix)

"turn one hot encoding to indices"
onehot2indices(xs::OneHotArray) = onehot2indices(xs.data)

#cpu onehot to indices
onehot2indices(x::AbstractArray{OneHotVector}) = map(i->Int(i.ix), x)

#cpu indices to onehot
indices2onehot(nums::Int, xs::AbstractArray{Int}) = map(i->onehot(i, 1:nums), xs)


import CuArrays: CuArray, cudaconvert
import Base.Broadcast: BroadcastStyle, ArrayStyle
BroadcastStyle(::Type{<:OneHotArray{N, <:CuArray} where N}) = ArrayStyle{CuArray}()
cudaconvert(x::OneHotArray{<:CuArray}) = OneHotArray(x.dims, cudaconvert(x.data))

_labelindex(num::Int, x::AbstractArray) = x .+ (num << 32)


#gpu indices to onehot
indices2onehot(num::Int, xs::CuArray{Int}) = convert(CuArray{OneHotVector}, _labelindex(num, xs))

#gpu onehot to indices
onehot2indices(x::CuArray{OneHotVector}) = convert(CuArray{Int}, x)


import Base: *
*(A::AbstractMatrix{T}, b::OneHotArray{N}) where {T,N} = invoke(gather, Tuple{AbstractMatrix, OneHotArray}, A, b)

*(A::TrackedMatrix{T}, b::OneHotArray{N}) where {T,N} = invoke(gather, Tuple{TrackedMatrix, OneHotArray}, A, b)


# ca = randn(512,  40000)
# cb = OneHotArray(40000, [1,3,2,1,11,12,13, 15])

# ca = randn(512,  40000)
# cb = OneHotArray(40000, ones(Int, 20))


# ga = cu(ca)
# gb = cu(cb)

# using Flux: back!, param
# pca = param(ca)
# pga = param(ga)

