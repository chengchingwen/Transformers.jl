using Transformers
using Transformers.Basic
using Test
using Random

import Flux
using Flux: gradient

using CUDA

const tests = [
    "tokenizer",
    "huggingface",
]

Random.seed!(0)

@testset "Transformers" begin
    for t in tests
        name = titlecase(t)
        @testset "$name" begin
            @info "Test $name"
            for f ∈ readdir(joinpath(@__DIR__, t))
                endswith(f, ".jl") && include(joinpath(@__DIR__, t, f))
            end
        end
    end
end
