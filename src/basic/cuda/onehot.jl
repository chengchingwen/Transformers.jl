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


OneHotArray(xs::A) where A = OneHotArray{ndims(A)+1, A}(Int(xs[1].of), xs)
OneHotArray(nums::Int, xs::AbstractArray{Int}) = OneHotArray(nums, map(i->onehot(i, 1:nums), xs))

import Adapt: adapt, adapt_structure

adapt_structure(T, xs::OneHotArray) = OneHotArray(xs.dims, adapt(T, xs.data))

"turn one hot encoding to indices"
onehot2indices(xs::OneHotArray) = onehot2indices(xs.data)

#cpu onehot to indices
function onehot2indices(x::AbstractArray{OneHotVector})
    ys = similar(x, Int, size(x)...)

    Threads.@threads for i = 1:length(x)
        @inbounds ys[i] = x[i].ix
    end

    return ys
end


import CuArrays: CuArray, cudaconvert
import Base.Broadcast: BroadcastStyle, ArrayStyle
BroadcastStyle(::Type{<:OneHotArray{N, <:CuArray} where N}) = ArrayStyle{CuArray}()
cudaconvert(x::OneHotArray{<:CuArray}) = OneHotArray(x.dims, cudaconvert(x.data))


# ca = randn(512,  40000)
# cb = OneHotArray(40000, [1,3,2,1,11,12,13, 15])

# ca = randn(512,  40000)
# cb = OneHotArray(40000, ones(Int, 20))


# ga = cu(ca)
# gb = cu(cb)

# using Flux: back!, param
# pca = param(ca)
# pga = param(ga)

#gpu onehot to indices
function onehot2indices(x::CuArray{OneHotVector})
    ys = CuArray{Int}(undef, size(x)...)

    function kernel!(ys, x)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

        if i <= length(x)
            ys[i] = x[i].ix
        end

        return
    end
    thr = min(256, length(x))
    blk = ceil.(Int, length(x) / thr)
    @cuda blocks=blk threads=thr kernel!(ys, x)
    return ys
end



include("./gather.jl")


import Base: *
*(A::AbstractMatrix{T}, b::OneHotArray{N}) where {T,N} = invoke(gather, Tuple{AbstractMatrix, OneHotArray}, A, b)

*(A::TrackedMatrix{T}, b::OneHotArray{N}) where {T,N} = invoke(gather, Tuple{TrackedMatrix, OneHotArray}, A, b)
