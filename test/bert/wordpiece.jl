@testset "WordPiece" begin
  wp1 = WordPiece(["[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn", "##ing", ","])

  @test wp1(bert_uncased_tokenizer("UNwant\u00E9d,running")) == ["un", "##want", "##ed", ",", "runn", "##ing"]
  @test wp1(Int, bert_uncased_tokenizer("UNwant\u00E9d,running")) == [8, 5, 6, 11, 9, 10]

  wp2 = WordPiece(["[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn", "##ing"])

  @test wp2(bert_uncased_tokenizer("")) == []
  @test wp2(bert_uncased_tokenizer("unwanted running")) == ["un", "##want", "##ed", "runn", "##ing"]
  @test wp2(bert_uncased_tokenizer("unwantedX running")) == ["[UNK]", "runn", "##ing"]

  vocab = Vocabulary(wp2)
  @test vocab(["un", "##want", "##ed", "runn", "##ing"]) == [8, 5, 6, 9, 10]
end
