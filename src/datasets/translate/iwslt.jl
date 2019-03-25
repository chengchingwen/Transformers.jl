module IWSLT
using DataDeps

using ..Datasets: Dataset
import ..Datasets: testfile, devfile, trainfile
import ..Datasets: token_freq, get_vocab

export IWSLT2016

function __init__()
    iwslt2016_init()
end

include("./iwslt2016.jl")


end
