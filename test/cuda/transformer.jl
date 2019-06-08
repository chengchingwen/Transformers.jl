@testset "Transformer" begin
    import Flux: gpu
    t = Transformer(10, 3, 15, 20) |> gpu
    td = TransformerDecoder(10, 3, 15, 20) |> gpu
    x = cu(randn(10, 7, 3))
    y = cu(randn(10, 6, 3))

    @test size(t(x)) == (10, 7, 3)
    @test size(t(x[:, :, 2])) == (10, 7)

    @test size(td(y, x)) == (10, 6, 3)
    @test size(td(y[:,:,2], x[:,:,2])) == (10, 6)
end
