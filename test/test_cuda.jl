using CuArrays

@testset "CUDA" begin
    CuArrays.allowscalar(false)
    @info "Testing CUDA"
    for f ∈ readdir("./cuda/")
        include("./cuda/$f")
    end
end
