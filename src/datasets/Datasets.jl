module Datasets

using DataDeps
using HTTP
using WordTokenizers
using Fetch

using ..Transformers: Container

export Dataset, Train, Dev, Test
export dataset, datafile, get_batch, get_vocab, get_labels, batched


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
