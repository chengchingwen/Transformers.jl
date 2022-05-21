using CUDA

import CUDA: CuArray

@generated function togpudevice(x::T) where {T <: AbstractArray}
    # https://github.com/FluxML/Flux.jl/blob/master/src/functor.jl#L89
    _R = Core.Compiler.return_type(CUDA.cu, Tuple{x})
    R = if _R isa Union
        T == _R.a ? _R.b : _R.a
    else
        _R
    end
    return :(gpu(x)::$R)
end

togpudevice(x) = gpu(x)
togpudevice(x, xs...) = (togpudevice(x), map(togpudevice, xs)...)

include("./statistic.jl")
include("./batch_gemm_gpu.jl")
include("./batch_tril_gpu.jl")
