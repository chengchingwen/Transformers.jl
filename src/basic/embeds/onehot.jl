using PrimitiveOneHot
using PrimitiveOneHot: OneHot, onehotsize

# old utility

onehot2indices(xs::OneHotArray) = reinterpret(Int32, xs.onehots)

indices2onehot(nums::Int, xs::AbstractArray{Int}) = OneHotArray(nums, xs)

tofloat(::Type{F}, o::OneHotArray{N, <:AbstractArray}) where {F<:AbstractFloat,N} = Array{F}(o)

# AD
using Flux
Flux.@nograd onehot2indices
Flux.@nograd indices2onehot
Flux.@nograd tofloat
