module Transformers

using Flux

export Transformer, TransformerDecoder
export Stack, stack, @nntopo_str, show_stackfunc

export dataset, datafile, get_batch, get_vocab

export Gpt, load_gpt_pretrain, lmloss

const Abstract3DTensor{T} = AbstractArray{T, 3}
const Container{T} = Union{NTuple{N, T}, Vector{T}} where N

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
