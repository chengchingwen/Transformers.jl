module GenerativePreTrain

using Flux

using ..Transformers: Abstract3DTensor
using ..Basic
export GPTTextEncoder, GPT2TextEncoder

include("./tokenizer.jl")
include("./textencoder.jl")

end
