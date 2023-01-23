using Transformers
using Test
using Random

import Flux
using Flux: gradient

using CUDA

function should_test_cuda()
    e = get(ENV, "JL_PKG_TEST_CUDA", false)
    e isa Bool && return e
    if e isa String
        x = tryparse(Bool, e)
        return isnothing(x) ? false : x
    else
        return false
    end
end

const USE_CUDA = @show should_test_cuda()

if USE_CUDA
    CUDA.allowscalar(false)
end

device(x) = USE_CUDA ? gpu(x) : x

drandn(arg...) = randn(arg...) |> device
drand(arg...) = rand(arg...) |> device
dones(arg...) = ones(arg...) |> device
dzeros(arg...) = zeros(arg...) |> device

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
    include("loss.jl")
end
