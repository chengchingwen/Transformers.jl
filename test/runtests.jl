using CondaPkg
testproj_dir = dirname(Base.load_path()[1])
cp(joinpath(@__DIR__, "CondaPkg.toml"), joinpath(testproj_dir, "CondaPkg.toml"))

using Transformers
using Test
using Random

import Flux
using Flux: gradient

using CUDA

const HFHUB_Token = get(ENV, "HUGGINGFACEHUB_TOKEN", nothing)

function envget(var)
    e = get(ENV, var, false)
    e isa Bool && return e
    if e isa String
        x = tryparse(Bool, e)
        return isnothing(x) ? false : x
    else
        return false
    end
end
should_test_cuda() = envget("JL_PKG_TEST_CUDA")

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
    "loss.jl",
    "grad.jl",
]

Random.seed!(0)

@testset "Transformers" begin
    for t in tests
        path = joinpath(@__DIR__, t)
        if isdir(path)
            name = titlecase(t)
            @testset "$name" begin
                @info "Test $name"
                for f ∈ readdir(path)
                    endswith(f, ".jl") || continue
                    t == "huggingface" && f == "tokenizer.jl" && !envget("JL_TRF_TEST_TKR") && continue
                    include(joinpath(path, f))
                end
            end
        else
            name = titlecase(first(splitext(t)))
            @info "Test $name"
            include(path)
        end
    end
end
