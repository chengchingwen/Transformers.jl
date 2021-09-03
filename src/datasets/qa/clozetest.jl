module ClozeTest
using Fetch
using DataDeps

using ..Datasets: Dataset
import ..Datasets: testfile, trainfile, get_labels

export StoryCloze

function __init__()
    storycloze_init()
end

include("./storycloze.jl")


end
