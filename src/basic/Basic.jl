module Basic

using Flux
using Flux: @functor
using Requires
using Requires: @init

using PrimitiveOneHot
import NNlib: gather

using ..Transformers: Abstract3DTensor, Container, epsilon

export OneHotArray, OneHot
export TransformerTextEncoder, encode, decode, lookup

include("./embeds/Embeds.jl")
include("./loss.jl")

end
