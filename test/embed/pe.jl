@testset "PositionEmbedding" begin

    pe = PositionEmbedding(10, 10)

    @test pe(randn(3, 5)) == pe.embedding[:, 1:5]
    @test pe(randn(3, 5, 7)) == pe.embedding[:, 1:5]
    @test pe(randn(3, 15, 2)) == pe.embedding[:, 1:15]
end
