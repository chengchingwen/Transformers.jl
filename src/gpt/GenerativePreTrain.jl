module GenerativePreTrain

using Flux

using ..Transformers: device, ThreeDimArray, gelu, matmul
using ..Basic

include("./gpt.jl")
include("./load_pretrain.jl")

end
