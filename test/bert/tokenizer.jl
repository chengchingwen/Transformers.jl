@testset "Tokenizer" begin
  @test bert_uncased_tokenizer(" \tHeLLo!how  \n Are yoU?  ") == ["hello", "!", "how", "are", "you", "?"]
  @test bert_uncased_tokenizer("H\u00E9llo") == ["hello"]
  @test bert_uncased_tokenizer("ah\u535A\u63A8zz") == ["ah", "\u535A", "\u63A8", "zz"]

  @test bert_cased_tokenizer(" \tHeLLo!how  \n Are yoU?  ") == ["HeLLo", "!", "how", "Are", "yoU", "?"]

  #check space definition consistent with bert
  @test isspace(' ')
  @test isspace('\t')
  @test isspace('\r')
  @test isspace('\n')
  @test isspace('\u00A0')
  @test !isspace('A')
  @test !isspace('-')

  @test BidirectionalEncoder.isinvalid('\u0005')
  @test !BidirectionalEncoder.isinvalid('A')
  @test !BidirectionalEncoder.isinvalid(' ')
  @test !BidirectionalEncoder.isinvalid('\t')
  @test !BidirectionalEncoder.isinvalid('\r')
  @test !BidirectionalEncoder.isinvalid('\U0001F4A9')

  @test BidirectionalEncoder.isbertpunct('-')
  @test BidirectionalEncoder.isbertpunct('$')
  @test BidirectionalEncoder.isbertpunct('`')
  @test BidirectionalEncoder.isbertpunct('.')
  @test !BidirectionalEncoder.isbertpunct('A')
  @test !BidirectionalEncoder.isbertpunct(' ')
end
