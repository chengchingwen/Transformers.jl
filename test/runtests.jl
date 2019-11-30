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

if v"1.0.0" <= VERSION  <= v"1.2.0"
    @test isempty(detect_ambiguities(Transformers, Transformers.GenerativePreTrain, Transformers.Basic, Transformers.Datasets))
end

const test_gpu = Base.find_package("CuArrays") !== nothing

@testset "Transformers" begin
    if test_gpu
        @info "Test CUDA"
        include("test_cuda.jl")
    else
        for t in tests
            fp = joinpath(dirname(@__FILE__), "test_$t.jl")
            @info "Test $(uppercase(t))"
            include(fp)
        end
    end
end
