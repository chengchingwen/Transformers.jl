module Transformers

using Flux

export device, use_gpu
export Transformer, TransformerDecoder
export Stack, stack, @nntopo_str, show_stackfunc

export dataset, datafile, get_batch, get_vocab

export Gpt, load_gpt_pretrain, lmloss

const ThreeDimArray{T} = AbstractArray{T, 3}
const TwoDimArray{T} = AbstractArray{T, 2}
const Container{T} = Union{NTuple{N, T}, Vector{T}} where N

device(x) = cpu(x)
function use_gpu(use::Bool)
    if use
        @eval using CuArrays
        @eval device(x) = gpu(x)
    else
        @eval device(x) = cpu(x)
    end
end

#implement batchmul for flux
include("./fix/batchedmul.jl")

#matmul for convenient and little performance gain
include("./fix/matmul.jl")

#implement of gelu for gpu
include("./fix/gelu.jl")

#implement of one for gpu
include("./fix/one.jl")

#dropout noise shape impl
include("./fix/dropout.jl")

include("./basic/Basic.jl")
include("./datasets/Datasets.jl")

include("./gpt/GenerativePreTrain.jl")

using .Basic
using .Datasets
using .GenerativePreTrain

end # module
