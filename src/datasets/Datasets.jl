module Datasets

using DataDeps
using HTTP
using InternedStrings
using WordTokenizers

using ..Transformers: Container

export Dataset, Train, Dev, Test
export dataset, datafile, get_batch, get_vocab, get_labels, batched


include("download_utils.jl")
include("./dataset.jl")

include("translate/wmt.jl")
using .WMT

include("translate/iwslt.jl")
using .IWSLT

include("qa/clozetest.jl")
using .ClozeTest

include("glue/glue.jl")
using .GLUE

end
