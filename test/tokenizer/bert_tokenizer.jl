@testset "Tokenizer" begin
    using Transformers.TextEncoders
    using Transformers.TextEncoders: bert_uncased_tokenizer, bert_cased_tokenizer
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

    @test TextEncoders.isinvalid('\u0005')
    @test !TextEncoders.isinvalid('A')
    @test !TextEncoders.isinvalid(' ')
    @test !TextEncoders.isinvalid('\t')
    @test !TextEncoders.isinvalid('\r')
    @test !TextEncoders.isinvalid('\U0001F4A9')

    @test TextEncoders.isbertpunct('-')
    @test TextEncoders.isbertpunct('$')
    @test TextEncoders.isbertpunct('`')
    @test TextEncoders.isbertpunct('.')
    @test !TextEncoders.isbertpunct('A')
    @test !TextEncoders.isbertpunct(' ')
end
