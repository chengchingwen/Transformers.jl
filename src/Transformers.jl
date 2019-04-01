module Transformers

using Flux
using Requires
using Requires: @init

export Transformer, TransformerDecoder
export Stack, stack, @nntopo_str, show_stackfunc

export dataset, datafile, get_batch, get_vocab

export todevice
export Gpt, load_gpt_pretrain, lmloss

const Abstract3DTensor{T} = AbstractArray{T, 3}
const Container{T} = Union{NTuple{N, T}, Vector{T}} where N

"move data to device, when CuArrays is loaded, basically = `CuArrays.cu` except `AbstractArray{Int}` become `CuArray{Int}`"
todevice(x) = x
todevice(x, xs...) = (x, xs...)

@init @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
    import .CuArrays

    "move data to device, basically = `CuArrays.cu` except `AbstractArray{Int}` become `CuArray{Int}`"
    todevice(x, xs...) = (todevice(x), todevice.(xs)...)
    todevice(x::AbstractArray{Int}) = CuArray(x)
    todevice(x) = CuArrays.cu(x)
end

#implement batchmul for flux
include("./fix/batchedmul.jl")

#implement of gelu for gpu
include("./fix/gelu.jl")

#dropout noise shape impl
include("./fix/dropout.jl")

include("./basic/Basic.jl")
include("./datasets/Datasets.jl")

include("./gpt/GenerativePreTrain.jl")

using .Basic
using .Datasets
using .GenerativePreTrain

end # module
