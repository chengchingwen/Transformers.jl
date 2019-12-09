using CuArrays
using CUDAnative

togpudevice(x) = gpu(x)
togpudevice(x, xs...) = (todevice(x), map(todevice, xs)...)
togpudevice(x::Union{Tuple, NamedTuple}) = map(todevice, x)
togpudevice(x::AbstractArray{Int}) = CuArrays.CuArray(x)

include("./scatter_gpu.jl")
include("./statistic.jl")
include("./batch_gemm_gpu.jl")
include("./gather_gpu.jl")
include("./onehot_gpu.jl")
