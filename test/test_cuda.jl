using CuArrays

@testset "CUDA" begin
    CuArrays.allowscalar(false)
    enable_gpu(true)
    @info "Testing CUDA"
    for f ∈ readdir("./cuda/")
        include("./cuda/$f")
    end
end
