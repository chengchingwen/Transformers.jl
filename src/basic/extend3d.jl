import Flux: logsoftmax, Dense

#extend Flux op for 3-dims input
function (a::LayerNorm)(x::ThreeDimArray{T}) where T
    s = size(x)
    reshape(a(reshape(x, s[1], :)), s)
end

function (d::Dense)(x::ThreeDimArray{T}) where T
    s = size(x)
    reshape(d(reshape(x, s[1], :)), size(d.W, 1), s[2], s[3])
end

#avoid ambiguity
(a::Dense{<:Any,W})(x::ThreeDimArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{ThreeDimArray}, x)

(a::Dense{<:Any,W})(x::ThreeDimArray{<:Real}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))


logsoftmax3d(x) = logsoftmax(x)
function logsoftmax3d(x::ThreeDimArray{T}) where T
    s = size(x)
    reshape(logsoftmax(reshape(x, s[1], :)), s)
end
