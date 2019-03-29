module Basic

using Flux
using ..Transformers: device,
    Abstract3DTensor, Container,
    batchedmul

export PositionEmbedding, Embed, getmask
export Transformer, TransformerDecoder

export NNTopo, @nntopo_str, @nntopo
export Stack, show_stackfunc, stack

export @toNd

export logkldivergence, logcrossentropy, logsoftmax3d

include("./extend3d.jl")
include("./position_embed.jl")
include("./mh_atten.jl")
include("./transformer.jl")
include("./topology.jl")
include("./stack.jl")
include("./cuda/embed.jl")
include("./loss.jl")



end
