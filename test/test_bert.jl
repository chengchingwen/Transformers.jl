@testset "Bert" begin
  using Transformers.BidirectionalEncoder
  using Transformers.BidirectionalEncoder: bert_cased_tokenizer, bert_uncased_tokenizer, WordPiece
  for f ∈ readdir("./bert/")
    include("./bert/$f")
  end
end
