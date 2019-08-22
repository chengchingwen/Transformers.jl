using Transformers
using Transformers.Basic
using Test

import Flux

const tests = [
    "transformer",
    "nntopo",
    "embed",
    "basic",
    "gpt",
    "bert",
    "pretrain",
]

if v"1.0.0" <= VERSION  <= v"1.1.0"
    @test isempty(detect_ambiguities(Transformers, Transformers.GenerativePreTrain, Transformers.Basic, Transformers.Datasets))
end

@testset "Transformers" begin
    for t in tests
        fp = joinpath(dirname(@__FILE__), "test_$t.jl")
        @info "Test $(uppercase(t))"
        include(fp)
    end

    if Base.find_package("CuArrays") != nothing
        include("test_cuda.jl")
    end
end
