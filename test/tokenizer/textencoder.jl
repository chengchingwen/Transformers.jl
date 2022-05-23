using Test
using BSON
using Transformers.Basic
using Transformers.BidirectionalEncoder


@testset "TextEncoder" begin
    @testset "Transformer" begin
        startsym = "11"
        endsym = "12"
        unksym = "0"
        labels = [unksym, startsym, endsym, collect(map(string, 1:10))...]
        textenc = TransformerTextEncoder(split, labels; startsym, endsym, unksym, padsym = unksym, trunc = 10)

        gen_data(n) = join(rand(1:10, n), ' ')

        # single input below trunc
        d1 = gen_data(7)
        s1 = [startsym; split(d1); endsym]
        @test decode(textenc, encode(textenc, d1).tok) == s1
        e1 = encode(textenc, d1)
        @test reinterpret(Int32, e1.tok) == map(x->findfirst(==(x), labels), s1)
        @test e1.mask == nothing
        # single input over trunc
        d2 = gen_data(15)
        s2 = [startsym; split(d2); endsym][begin:10]
        @test decode(textenc, encode(textenc, d2).tok) == s2
        e2 = encode(textenc, d2)
        @test reinterpret(Int32, e2.tok) == map(x->findfirst(==(x), labels), s2)
        @test e2.mask == nothing
        # batch input below trunc
        d3 = [gen_data(5), gen_data(7)]
        s3 = [startsym; split(d3[1]); endsym; unksym; unksym;; startsym; split(d3[2]); endsym]
        @test decode(textenc, encode(textenc, d3).tok) == s3
        e3 = encode(textenc, d3)
        @test reinterpret(Int32, e3.tok) == map(x->findfirst(==(x), labels), s3)
        @test e3.mask == reshape([fill(1.0, 7); 0.;0.;; fill(1.0, 9)], (1, 9, 2))
        # batch input over trunc
        d4 = [gen_data(7), gen_data(12)]
        s4 = [startsym; split(d4[1]); endsym; unksym;; startsym; split(d4[2])[begin:9]]
        @test decode(textenc, encode(textenc, d4).tok) == s4
        e4 = encode(textenc, d4)
        @test reinterpret(Int32, e4.tok) == map(x->findfirst(==(x), labels), s4)
        @test e4.mask == reshape([fill(1.0, 9); 0.;; fill(1.0, 10)], (1, 10, 2))
    end

    @testset "Bert" begin
        BSON.@load "$(joinpath(@__DIR__, "wordpiece.bson"))" wordpiece
        bertenc = BertTextEncoder(bert_cased_tokenizer, wordpiece; trunc=15)
        d1 = "Peter Piper picked a peck of pickled peppers"
        s1 = ["[CLS]", "Peter", "Piper", "picked", "a", "p", "##eck", "of", "pick", "##led", "pepper", "##s", "[SEP]"]
        d2 = "Fuzzy Wuzzy was a bear"
        s2 = ["[CLS]", "Fu", "##zzy", "Wu", "##zzy", "was", "a", "bear", "[SEP]"]

        @test decode(bertenc, encode(bertenc, d1).input.tok) == s1
        e1 = encode(bertenc, d1)
        @test reinterpret(Int32, e1.input.tok) == map(x->findfirst(==(x), bertenc.vocab.list), s1)
        @test e1.input.segment == ones(Int, length(s1))
        @test e1.mask == nothing
        d3 = [d2, d1]
        s3 = [s2; fill(bertenc.vocab.unk, 4);; s1]
        @test decode(bertenc, encode(bertenc, d3).input.tok) == s3
        e3 = encode(bertenc, d3)
        @test reinterpret(Int32, e3.input.tok) == map(x->findfirst(==(x), bertenc.vocab.list), s3)
        @test e3.input.segment == ones(Int, 13, 2)
        @test e3.mask == [[ones(Float32, length(s2)); zeros(Float32, length(s1)-length(s2))]' ;;; ones(Int, length(s1))']
        d4 = [[d2, d1], [d2,]]
        s4 = [s2; s1[begin+1:(16-length(s2))] ;; s2; fill(bertenc.vocab.unk, 15-length(s2))]
        @test decode(bertenc, encode(bertenc, d4).input.tok) == s4
        e4 = encode(bertenc, d4)
        @test reinterpret(Int32, e4.input.tok) == map(x->findfirst(==(x), bertenc.vocab.list), s4)
        @test e4.input.segment == [ones(Int, length(s2)); fill(2, 15-length(s2)) ;; ones(Int, 15)]
        @test e4.mask == [ones(Float32, 1, 15) ;;; [ones(Float32, 1, length(s2)) ;; zeros(Float32, 1, 15-length(s2))]]
    end
end
