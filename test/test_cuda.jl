using CuArrays

@testset "CUDA" begin
    @info "Testing CUDA"
    for f ∈ readdir("./cuda/")
        include("./cuda/$f")
    end
end
