module ClozeTest
using DataDeps

using ..Datasets: Dataset, maybegoogle_download, get_labels
import ..Datasets: testfile, trainfile

export StoryCloze

function __init__()
    storycloze_init()
end

include("./storycloze.jl")


end
