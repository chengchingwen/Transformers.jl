module WMT
using Fetch
using DataDeps
using BytePairEncoding

using ..Datasets: Dataset
import ..Datasets: testfile, trainfile, get_vocab

export GoogleWMT

function __init__()
    googlewmt_init()
end

include("./google_wmt.jl")


end
