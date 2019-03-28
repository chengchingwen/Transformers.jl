module GenerativePreTrain

using Flux

using ..Transformers: device, Abstract3DTensor
using ..Basic

include("./gpt.jl")
include("./load_pretrain.jl")

end
