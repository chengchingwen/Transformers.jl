module WMT
using DataDeps
using InternedStrings
using BytePairEncoding

using ..Datasets: download_gdrive, Dataset
import ..Datasets: testfile, trainfile, get_vocab

export GoogleWMT

function __init__()
    googlewmt_init()
end

include("./google_wmt.jl")


end
