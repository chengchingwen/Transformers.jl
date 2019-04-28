@testset "Embed" begin
    for f ∈ readdir("./embed/")
        include("./embed/$f")
    end
end
