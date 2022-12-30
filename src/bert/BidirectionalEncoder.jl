module BidirectionalEncoder

using Flux

using TextEncodeBase

using ..Basic
export BertTextEncoder, bert_cased_tokenizer, bert_uncased_tokenizer

include("tokenizer.jl")
include("textencoder.jl")

end
