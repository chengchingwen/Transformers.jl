module Stacks

export NNTopo, @nntopo_str, @nntopo
export Stack, show_stackfunc, stack

include("topology.jl")
include("./stack.jl")

end
