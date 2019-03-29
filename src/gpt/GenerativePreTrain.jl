module GenerativePreTrain

using Flux

using ..Transformers: Abstract3DTensor
using ..Basic

include("./gpt.jl")
include("./load_pretrain.jl")

end
