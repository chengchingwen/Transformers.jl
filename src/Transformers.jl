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

using CuArrays
const use_cuda = Ref(false)


#implement batchmul for flux
include("./fix/batchedmul.jl")

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


function __init__()
  precompiling = ccall(:jl_generating_output, Cint, ()) != 0

  # we don't want to include the CUDA module when precompiling,
  # or we could end up replacing it at run time (triggering a warning)
  precompiling && return

  if !CuArrays.functional()
    # nothing to do here, and either CuArrays or one of its dependencies will have warned
  else
    use_cuda[] = true
    include(joinpath(@__DIR__, "cuda/cuda.jl"))
  end
end


end # module
