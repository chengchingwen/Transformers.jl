@testset "EmbedOnehot" begin
    using Flux: data
    multi(x, n) = collect(Iterators.repeated(x, n))

    after_enc = [2,2,4,5,6,1,1]
    multi_after_enc = hcat(multi(after_enc, 5)...)

    e = Embed(10, 6)

    @test data(e(after_enc)) == data(hcat(map(i->e.embedding[:,i], after_enc)...))
    @test eltype(data(e(after_enc, 0.5))) == Float32

    @test size(e) == (10, 6)
    @test size(e, 1) == 10
    @test size(e, 2) == 6
end
