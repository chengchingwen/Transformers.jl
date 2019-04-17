@testset "Embed" begin
    using Flux: onehot, onecold, data
    multi(x, n) = collect(Iterators.repeated(x, n))

    v = Vocabulary(["a", "b", "c", "d", "e"], "?")

    before_enc = ["a", "a", "c" ,"d", "e","x", "sad"]
    before_unk_enc = ["a", "a", "c" ,"d", "e","?", "?"]
    after_enc = [2,2,4,5,6,1,1]
    multi_after_enc = hcat(multi(after_enc, 5)...)

    e = Embed(10, v)

    @test data(e(after_enc)) == data(hcat(map(i->e.embedding[:,i], after_enc)...))
    @test data(e(before_enc)) == data(hcat(map(x->e.embedding[:, v(x)], before_unk_enc)...))
    @test onehot(e, before_enc) == onehotarray(length(v), v(before_unk_enc))
    @test onehot(e, after_enc) == onehotarray(length(v), after_enc)

    @test onehot(e, multi(before_enc,5)) == onehotarray(length(v), v(multi(before_unk_enc,5)))
    @test onehot(e, multi_after_enc) == onehotarray(length(v), multi_after_enc)

    @test onecold(e, onehot(e, multi(before_enc,5))) == multi(before_unk_enc,5)
    @test onecold(e, onehot(e, multi_after_enc)) == multi(before_unk_enc,5)
    @test onecold(e, onehot(e, before_enc)) == before_unk_enc
    @test onecold(e, onehot(e, after_enc)) == before_unk_enc
end
