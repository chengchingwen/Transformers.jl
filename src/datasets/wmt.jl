module WMT
using DataDeps
using InternedStrings
using BytePairEncoding

using ..Datasets: download_gdrive, Dataset
import ..Datasets: testfile, trainfile, get_vocab

export GoogleWMT

include("./google_wmt.jl")


end
