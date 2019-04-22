@testset "Transformer" begin
    t = Transformer(10, 3, 15, 20)
    x = randn(10, 7, 3)

    @test size(t(x)) == (10, 7, 3)
    @test size(t(x[:, :, 2])) == (10, 7)
end
