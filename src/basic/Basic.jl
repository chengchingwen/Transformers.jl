module Basic

using Flux
using Requires
using Requires: @init

using PrimitiveOneHot
import NNlib: gather

using ..Transformers: Abstract3DTensor, Container, epsilon, batchedmul, batched_triu!

export CompositeEmbedding, TransformerModel, set_classifier, clear_classifier
export PositionEmbedding, Embed, getmask,
    Vocabulary, gather
export OneHotArray, indices2onehot, onehot2indices,
    onehot, tofloat
export TransformerTextEncoder, encode, decode, lookup
export Transformer, TransformerDecoder

export @toNd, Positionwise

export logkldivergence, logcrossentropy

include("./extend3d.jl")
include("./embeds/Embeds.jl")
include("./mh_atten.jl")
include("./transformer.jl")
include("./loss.jl")
include("./model.jl")

end
