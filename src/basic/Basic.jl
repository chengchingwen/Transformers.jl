module Basic

using Flux
using ..Transformers: device, get_ftype,
    ThreeDimArray, TwoDimArray, Container,
    batchedmul, matmul

export PositionEmbedding, Embed, getmask
export Transformer, TransformerDecoder

export NNTopo, @nntopo_str, @nntopo
export Stack, show_stackfunc, stack

export @toNd

export logkldivergence, logcrossentropy, logsoftmax3d

include("./position_embed.jl")
include("./mh_atten.jl")
include("./transformer.jl")
include("./topology.jl")
include("./stack.jl")
include("./embed.jl")
include("./loss.jl")
include("./extend3d.jl")


end
