@testset "WordPiece" begin
    using TextEncodeBase
    using Transformers.WordPieceModel
    wp1 = WordPiece(["[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn", "##ing", ","])

    @test mapreduce(wp1, append!, bert_uncased_tokenizer("UNwant\u00E9d,running")) ==
        ["un", "##want", "##ed", ",", "runn", "##ing"]

    wp2 = WordPiece(["[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn", "##ing"])

    @test mapreduce(wp2, append!, bert_uncased_tokenizer(""); init = []) == []
    @test mapreduce(wp2, append!, bert_uncased_tokenizer("unwanted running")) ==
        ["un", "##want", "##ed", "runn", "##ing"]
    @test mapreduce(wp2, append!, bert_uncased_tokenizer("unwantedX running")) ==
        ["[UNK]", "runn", "##ing"]

    vocab = Vocab(wp2)
    @test lookup(vocab, ["un", "##want", "##ed", "runn", "##ing"]) == [8, 5, 6, 9, 10]
end
