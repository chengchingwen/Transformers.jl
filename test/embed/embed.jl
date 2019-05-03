@testset "EmbedOnehot" begin
    using Flux: onehot, onecold, data
    multi(x, n) = collect(Iterators.repeated(x, n))

    v = Vocabulary(["a", "b", "c", "d", "e"], "?")

    before_enc = ["a", "a", "c" ,"d", "e","x", "sad"]
    before_unk_enc = ["a", "a", "c" ,"d", "e","?", "?"]
    after_enc = [2,2,4,5,6,1,1]
    multi_after_enc = hcat(multi(after_enc, 5)...)

    e = Embed(10, length(v))

    @test data(e(after_enc)) == data(hcat(map(i->e.embedding[:,i], after_enc)...))

    @test onehot(v, before_enc) == onehotarray(length(v), v(before_unk_enc))
    @test onehot(v, after_enc) == onehotarray(length(v), after_enc)
    @test onehot(v, multi(before_enc,5)) == onehotarray(length(v), v(multi(before_unk_enc,5)))
    @test onehot(v, multi_after_enc) == onehotarray(length(v), multi_after_enc)

    @test onecold(v, onehot(v, multi(before_enc,5))) == multi(before_unk_enc,5)
    @test onecold(v, onehot(v, multi_after_enc)) == multi(before_unk_enc,5)
    @test onecold(v, onehot(v, before_enc)) == before_unk_enc
    @test onecold(v, onehot(v, after_enc)) == before_unk_enc

    @test eltype(data(e(after_enc, 0.5))) == Float32

    @test size(e) == (10, 6)
    @test size(e, 1) == 10
    @test size(e, 2) == 6 == length(v)
end
