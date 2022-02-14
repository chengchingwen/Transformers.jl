using CUDA

import CUDA: CuArray

togpudevice(x) = gpu(x)
togpudevice(x, xs...) = (todevice(x), map(todevice, xs)...)

include("./statistic.jl")
include("./batch_gemm_gpu.jl")
include("./batch_tril_gpu.jl")
