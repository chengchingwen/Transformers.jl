module ClozeTest
using DataDeps

using ..Datasets: Dataset, maybegoogle_download
import ..Datasets: testfile, trainfile, get_labels

export StoryCloze

function __init__()
    storycloze_init()
end

include("./storycloze.jl")


end
