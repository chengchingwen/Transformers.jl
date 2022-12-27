using Transformers
using Transformers.Basic
using Test
using Random

import Flux
using Flux: gradient

using CUDA

const tests = [
    "bert",
]

Random.seed!(0)

@testset "Transformers" begin
    for t in tests
        fp = joinpath(@__DIR__, "test_$t.jl")
        @info "Test $(uppercase(t))"
        include(fp)
    end

    @info "Test text encoder"
    include(joinpath(@__DIR__, "tokenizer/textencoder.jl"))
end
