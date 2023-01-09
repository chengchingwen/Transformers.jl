module Transformers

using Flux
using Requires
using Requires: @init

using NeuralAttentionlib

export Transformer

export dataset, datafile, get_batch, get_vocab

export todevice, enable_gpu

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
        @eval todevice(args...) = togpudevice(args...)
    else
        @eval todevice(args...) = tocpudevice(args...)
    end
end

"""
  todevice(x)

move data to device, only when gpu is enable with `enable_gpu`, basically equal `Flux.gpu` except `AbstractArray{Int}` become `CuArray{Int}`. Otherwise equal `Flux.cpu`
"""
todevice(args...) = tocpudevice(args...)

tocpudevice(x) = cpu(x)
tocpudevice(x, xs...) = (tocpudevice(x), map(tocpudevice, xs)...)

@generated function tocpudevice(x::T) where T <: AbstractArray
    R = Core.Compiler.return_type(Flux.adapt, Tuple{Type{Array}, x})
    return :(cpu(x)::$R)
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
