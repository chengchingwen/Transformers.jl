import Flux: OneHotVector
import CUDA: cudaconvert
import Base.Broadcast: BroadcastStyle, ArrayStyle

import .Basic: indices2onehot, onehot2indices, tofloat, _labelindex

Base.similar(o::OneHotArray{N, <:CuArray}) where N = similar(o, size(o))
Base.similar(o::OneHotArray{N, <:CuArray}, dims::NTuple{N, Int}) where N = similar(o, Bool, dims)
Base.similar(o::OneHotArray{N, <:CuArray}, dims::Int...) where N = similar(o, Bool, Tuple(dims))
Base.similar(o::OneHotArray{N, <:CuArray}, T::Type) where N = similar(o, T, size(o))
Base.similar(o::OneHotArray{N, <:CuArray}, T::Type, dims::Int...) where N = similar(o, T, Tuple(dims))
Base.similar(o::OneHotArray{N, <:CuArray}, T::Type, dims::NTuple{N, Int}) where N = CuArray{T}(undef, dims)

tofloat(::Type{F}, o::OneHotArray{N, <:CuArray}) where {F<:AbstractFloat,N} = CuArray{F}(o)

BroadcastStyle(::Type{<:OneHotArray{N, <:CuArray} where N}) = ArrayStyle{CuArray}()
cudaconvert(x::OneHotArray{<:CuArray}) = OneHotArray(x.dims, cudaconvert(x.data))

#gpu indices to onehot
indices2onehot(num::Int, xs::CuArray{Int}) = convert(CuArray{OneHotVector}, _labelindex(num, xs))

#gpu onehot to indices
onehot2indices(x::CuArray{OneHotVector}) = convert(CuArray{Int}, x)
