module BidirectionalEncoder

using Flux
using Requires
using Requires: @init
using BSON

using TextEncodeBase

using ..Basic
using ..Basic: with_firsthead_tail, segment_and_concat
using ..Pretrain: isbson, iszip, istfbson, zipname, zipfile, findfile
export Bert, load_bert_pretrain, bert_pretrain_task, masklmloss
export BertTextEncoder, bert_cased_tokenizer, bert_uncased_tokenizer

include("bert.jl")
include("tfckpt2bson.jl")
include("load_pretrain.jl")
include("tokenizer.jl")
include("wordpiece.jl")
include("textencoder.jl")
include("pretrain_helper.jl")

end
