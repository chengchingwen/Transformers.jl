using Flux: @functor
import Flux
import Base

using TextEncodeBase

include("./utils.jl")
include("./tokenizer.jl")
include("./textencoder.jl")
include("./vocab.jl")
include("./onehot.jl")
include("./etype.jl")
include("./embed.jl")
include("./position_embed.jl")
