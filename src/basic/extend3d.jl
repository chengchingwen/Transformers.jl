import Flux: logsoftmax

#extend Flux op for 3-dims input
function (a::LayerNorm)(x::ThreeDimArray{T}) where T
    s = size(x)
    reshape(a(reshape(x, s[1], :)), s)
end

function (d::Dense)(x::ThreeDimArray{T}) where T
    s = size(x)
    reshape(d(reshape(x, s[1], :)), size(d.W, 1), s[2], s[3])
end

logsoftmax3d(x) = logsoftmax(x)
function logsoftmax3d(x::ThreeDimArray{T}) where T
    s = size(x)
    reshape(logsoftmax(reshape(x, s[1], :)), s)
end
