module Transformers

using Flux

using NeuralAttentionlib

export Transformer

export todevice, enable_gpu
export Layers, TextEncoders, HuggingFace

const Abstract3DTensor{T} = AbstractArray{T, 3}
const Container{T} = Union{NTuple{N, T}, Vector{T}} where N

using CUDA

"""
  enable_gpu(t=true)

enable gpu for `todevice`, disable with `enable_gpu(false)`.
"""
function enable_gpu(t::Bool=true)
    if t
        CUDA.functional() || error("CUDA not functional")
        @eval todevice(args...; kws...) = togpudevice(args...; kws...)
    else
        @eval todevice(args...; kws...) = tocpudevice(args...; kws...)
    end
end

"""
  todevice(x)

move data to device, only when gpu is enable with `enable_gpu`, basically equal `Flux.gpu` except `AbstractArray{Int}` become `CuArray{Int}`. Otherwise equal `Flux.cpu`
"""
todevice(args...; kws...) = tocpudevice(args...; kws...)

# https://github.com/FluxML/Flux.jl/blob/79971741ed8454cdf6a66515799a0c4b864f564a/src/functor.jl#L174
_tocpudevice(x, cache) = Flux.fmap(
    x -> Flux.adapt(Flux.FluxCPUAdaptor(), x),
    x; exclude = Flux._isleaf, cache)

function tocpudevice(x; cache = IdDict())
    # equivalent to Flux.cpu(x)
    return _tocpudevice(x, cache)
end
function tocpudevice(x, xs...; cache = IdDict())
    return (tocpudevice(x; cache), map(xi->tocpudevice(xi; cache), xs)...)
end
tocpudevice(x::Tuple; cache = IdDict()) = tocpudevice(x...; cache)
tocpudevice(x::NamedTuple{name}; cache = IdDict()) where name = NamedTuple{name}(tocpudevice(values(x)...; cache))

@generated function tocpudevice(x::T; cache = IdDict()) where {T <: Union{AbstractArray, NeuralAttentionlib.AbstractMask}}
    R = Core.Compiler.return_type(Flux.adapt, Tuple{Type{Array}, x})
    return :(_tocpudevice(x, cache)::$R)
end

include("./layers/Layers.jl")
include("./tokenizer/tokenizer.jl")
include("./textencoders/TextEncoders.jl")

include("./datasets/Datasets.jl")
include("./huggingface/HuggingFace.jl")

include("./loss.jl")
include("./cuda.jl")

using .Layers
using .TextEncoders
using .Datasets

using .HuggingFace

end # module
