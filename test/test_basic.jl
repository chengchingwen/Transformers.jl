@testset "Basic" begin

    d = Flux.Dense(10, 5)

    c = Flux.Chain(d, Flux.softmax)
    p = Positionwise(d, Flux.softmax)

    x = randn(10, 5, 3)

    @test p(x)[:,:,2] == c(x[:,:,2])
    @test p(x[:,:,2]) == c(x[:,:,2])
end
