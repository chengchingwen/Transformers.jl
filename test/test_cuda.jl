using CUDA

@testset "CUDA" begin
    CUDA.allowscalar(false)
    enable_gpu(true)
    for f ∈ readdir("./cuda/")
        include("./cuda/$f")
    end
end
