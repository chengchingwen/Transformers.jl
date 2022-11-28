module Transformers

using Flux
using Requires
using Requires: @init

export Transformer, TransformerDecoder
export Stack, @nntopo_str, @nntopo

export dataset, datafile, get_batch, get_vocab

export todevice, enable_gpu
export Gpt
export Bert

const Abstract3DTensor{T} = AbstractArray{T, 3}
const Container{T} = Union{NTuple{N, T}, Vector{T}} where N

const ϵ = Ref(1e-8)

"""
    set_ϵ(x)

set the ϵ value
"""
set_ϵ(x) = (ϵ[] = x; x)

"""
    epsilon(T)

get the ϵ value in type T
"""
epsilon(::Type{T}) where T = convert(T, ϵ[])

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

#implement batchmul, batchtril for flux
include("./fix/batchedmul.jl")
include("./fix/batched_tril.jl")

include("./basic/Basic.jl")
include("./stacks/Stacks.jl")
include("./datasets/Datasets.jl")

include("./pretrain/Pretrain.jl")

include("./tokenizer/tokenizer.jl")
include("./gpt/GenerativePreTrain.jl")
include("./bert/BidirectionalEncoder.jl")

include("./huggingface/HuggingFace.jl")

include("cuda/cuda.jl")

using .Basic
using .Stacks
using .Datasets
using .Pretrain
using .GenerativePreTrain
using .BidirectionalEncoder

using .HuggingFace

include("./experimental/experimental.jl")

end # module
