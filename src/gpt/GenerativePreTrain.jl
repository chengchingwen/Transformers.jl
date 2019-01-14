module GenerativePreTrain

using Flux

using ..Transformers: device, TwoDimArray, ThreeDimArray, gelu
using ..Basic

include("./gpt.jl")
include("./load_pretrain.jl")

end
