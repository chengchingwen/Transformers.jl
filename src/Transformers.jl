module Transformers

using Flux
using Requires
using Requires: @init

export Transformer, TransformerDecoder
export Stack, @nntopo_str, @nntopo

export dataset, datafile, get_batch, get_vocab

export todevice
export Gpt
export Bert

const Abstract3DTensor{T} = AbstractArray{T, 3}
const Container{T} = Union{NTuple{N, T}, Vector{T}} where N

"move data to device, when CuArrays is loaded, basically = `CuArrays.cu` except `AbstractArray{Int}` become `CuArray{Int}`"
todevice(x) = x
todevice(x, xs...) = (x, xs...)

@init @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
    import .CuArrays

    todevice(x) = CuArrays.cu(x)
    todevice(x, xs...) = (todevice(x), todevice.(xs)...)
    todevice(x::Union{Tuple, NamedTuple}) = map(todevice, x)
    todevice(x::AbstractArray{Int}) = CuArrays.CuArray(x)
end

#implement batchmul for flux
include("./fix/batchedmul.jl")

#dropout noise shape impl
include("./fix/dropout.jl")

#scatter/gather with atomic ops
include("./fix/atomic.jl")
include("./fix/scatter.jl")

include("./basic/Basic.jl")
include("./stacks/Stacks.jl")
include("./datasets/Datasets.jl")

include("./pretrain/Pretrain.jl")

include("./gpt/GenerativePreTrain.jl")
include("./bert/BidirectionalEncoder.jl")



using .Basic
using .Stacks
using .Datasets
using .Pretrain
using .GenerativePreTrain
using .BidirectionalEncoder


end # module
