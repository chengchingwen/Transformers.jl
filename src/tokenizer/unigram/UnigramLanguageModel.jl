module UnigramLanguageModel

using TextEncodeBase

export Unigram, UnigramTokenization

include("trie.jl")
include("dat.jl")
include("unigram.jl")
include("tokenization.jl")

end
