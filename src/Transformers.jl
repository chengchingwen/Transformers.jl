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

using CuArrays
const has_cuda = Ref(false)
const use_cuda = Ref(false)

"""
  enable_gpu(t=true)

enable gpu for todevice, disable with `enable_gpu(false)`.
"""
enable_gpu(t::Bool=true) = !has_cuda[] && t ? error("CuArrays not functional") : (use_cuda[] = t)

"""
  todevice(x)

move data to device, only when gpu is enable with `enable_gpu`, basically equal `Flux.gpu` except `AbstractArray{Int}` become `CuArray{Int}`. Otherwise equal `Flux.cpu`
"""
todevice(args...) = use_cuda[] ? togpudevice(args...) : tocpudevice(args...)

tocpudevice(x) = cpu(x)
tocpudevice(x, xs...) = (x, map(cpu, xs)...)
togpudevice(x...) = error("CuArrays not functional")

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
    has_cuda[] = true
    include(joinpath(@__DIR__, "cuda/cuda.jl"))
  end
end


end # module
