using Transformers
using Transformers.Basic
using Test

const tests = [
    "transformer",
    "embed",
]


@test isempty(detect_ambiguities(Transformers, Transformers.GenerativePreTrain, Transformers.Basic, Transformers.Datasets))

@testset "Transformers" begin
    for t in tests
        fp = joinpath(dirname(@__FILE__), "test_$t.jl")
        println("$fp ...")
        include(fp)
    end
end
