using Flux
using CUDA
using NeuralAttentionlib

function _togpudevice(x, cache)
    # https://github.com/FluxML/Flux.jl/blob/79971741ed8454cdf6a66515799a0c4b864f564a/src/functor.jl#L206-L209
    Flux.check_use_cuda()
    return Flux.fmap(
        x -> Flux.adapt(Flux.FluxCUDAAdaptor(), x),
        x; exclude = Flux._isleaf, cache)
end

togpudevice(x; cache = IdDict()) = _togpudevice(x, cache)
function togpudevice(x, xs...; cache = IdDict())
    return (togpudevice(x; cache), map(xi->togpudevice(xi; cache), xs)...)
end
togpudevice(x::Tuple; cache = IdDict()) = togpudevice(x...; cache)
togpudevice(x::NamedTuple{name}; cache = IdDict()) where name = NamedTuple{name}(togpudevice(values(x)...; cache))

@generated function togpudevice(x::T; cache = IdDict()) where {T <: Union{AbstractArray, NeuralAttentionlib.AbstractMask}}
    # https://github.com/FluxML/Flux.jl/blob/79971741ed8454cdf6a66515799a0c4b864f564a/src/functor.jl#L98
    _R = Core.Compiler.return_type(CUDA.cu, Tuple{x})
    R = if _R isa Union
        T == _R.a ? _R.b : _R.a
    else
        _R
    end
    return :(_togpudevice(x, cache)::$R)
end
