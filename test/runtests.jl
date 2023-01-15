using Transformers
using Transformers.Basic
using Test
using Random

import Flux
using Flux: gradient

using CUDA

const tests = [
    "transformer",
    "nntopo",
    "embed",
    "basic",
    "gpt",
    "bert",
    "clip",
]

Random.seed!(0)

if haskey(ENV, "TEST_TRANSFORMERS_PRETRAIN")
    push!(tests, "pretrain")
end

if CUDA.functional()
    # push!(tests, "cuda")
else
    @warn "CUDA unavailable, not testing GPU support"
end

@testset "Transformers" begin
    for t in tests
        fp = joinpath(@__DIR__, "test_$t.jl")
        @info "Test $(uppercase(t))"
        include(fp)
    end

    @info "Test text encoder"
    include(joinpath(@__DIR__, "tokenizer/textencoder.jl"))
end
