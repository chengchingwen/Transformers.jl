@testset "Vocabulary" begin
    using Flux: onehot, onecold
    multi(x, n) = collect(Iterators.repeated(x, n))

    v = Vocabulary(["a", "b", "c", "d", "e"], "?")

    before_enc = ["a", "a", "c" ,"d", "e","x", "sad"]
    before_unk_enc = ["a", "a", "c" ,"d", "e","?", "?"]
    after_enc = [2,2,4,5,6,1,1]
    multi_after_enc = hcat(multi(after_enc, 5)...)

    @test length(v) == 6

    @test v("a") == 2
    @test v[2] == "a"
    @test v(before_enc) == after_enc
    @test v[after_enc] == before_unk_enc
    @test v(multi(before_enc, 5)) == multi_after_enc
    @test v[multi_after_enc] == multi(before_unk_enc, 5)
    @test v[multi(after_enc, 5)] == multi(before_unk_enc, 5)

    @test encode(v, "a") == 2
    @test decode(v, 2) == "a"
    @test encode(v, before_enc) == after_enc
    @test decode(v, after_enc) == before_unk_enc
    @test encode(v, multi(before_enc, 5)) == multi_after_enc
    @test decode(v, multi_after_enc) == multi(before_unk_enc, 5)
    @test decode(v, multi(after_enc, 5)) == multi(before_unk_enc, 5)

    @test onehot(v, before_enc) == onehotarray(length(v), v(before_unk_enc))
    @test onehot(v, after_enc) == onehotarray(length(v), after_enc)
    @test onehot(v, multi(before_enc,5)) == onehotarray(length(v), v(multi(before_unk_enc,5)))
    @test onehot(v, multi_after_enc) == onehotarray(length(v), multi_after_enc)

    @test onecold(v, onehot(v, multi(before_enc,5))) == multi(before_unk_enc,5)
    @test onecold(v, onehot(v, multi_after_enc)) == multi(before_unk_enc,5)
    @test onecold(v, onehot(v, before_enc)) == before_unk_enc
    @test onecold(v, onehot(v, after_enc)) == before_unk_enc
end
