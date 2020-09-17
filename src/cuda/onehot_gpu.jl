import .Basic: tofloat

# old utility

tofloat(::Type{F}, o::OneHotArray) where {F<:AbstractFloat} = CuArray{F}(o)

# avoid scalar operation by broadcasting assigment
function CUDA.CuArray{F}(o::OneHotArray{K, N, var"N+1", A}) where {F <: AbstractFloat, K, N, var"N+1", A <: CuArray}
  dest = similar(o, F)
  dest .= o
  return dest
end

Base.BroadcastStyle(::Type{<: OneHotArray{K, N, var"N+1", A}}) where {K, N, var"N+1", A <: CuArray} = CUDA.CuArrayStyle{var"N+1"}()
