using Transformers
using Transformers.Basic
using Test
using Random

import Flux
using Flux: gradient

const tests = [
    "transformer",
    "nntopo",
    "embed",
    "basic",
    "gpt",
    "bert",
]

Random.seed!(0)

if v"1.0.0" <= VERSION  < v"1.4.0"
    @test isempty(detect_ambiguities(Transformers, Transformers.GenerativePreTrain, Transformers.Basic, Transformers.Datasets))
end

if haskey(ENV, "TEST_TRANSFORMERS_PRETRAIN")
    push!(tests, "pretrain")
end

@testset "Transformers" begin
  if Flux.use_cuda[]
    @info "Test CUDA"
    include("test_cuda.jl")
  else
    @warn "CUDA unavailable, not testing GPU support"
  end

  for t in tests
    fp = joinpath(dirname(@__FILE__), "test_$t.jl")
    @info "Test $(uppercase(t))"
    include(fp)
  end
end
