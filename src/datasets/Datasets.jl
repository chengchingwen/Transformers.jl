module Datasets

using DataDeps

using ..Transformers: Container

export Dataset, Train, Dev, Test
export dataset, datafile, get_batch, get_vocab


include("download_utils.jl")
include("./dataset.jl")

include("translate/wmt.jl")
using .WMT

include("translate/iwslt.jl")
using .IWSLT

include("qa/clozetest.jl")
using .ClozeTest


end
