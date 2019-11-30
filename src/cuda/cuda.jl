using CuArrays
using CUDAnative

todevice(x) = CuArrays.cu(x)
todevice(x, xs...) = (todevice(x), todevice.(xs)...)
todevice(x::Union{Tuple, NamedTuple}) = map(todevice, x)
todevice(x::AbstractArray{Int}) = CuArrays.CuArray(x)

include("./scatter_gpu.jl")
include("./statistic.jl")
include("./batch_gemm_gpu.jl")
include("./gather_gpu.jl")
include("./onehot_gpu.jl")
