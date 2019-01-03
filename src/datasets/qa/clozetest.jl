module ClozeTest
using DataDeps

using ..Datasets: Dataset, mybegoogle_download
import ..Datasets: testfile, trainfile

export StoryCloze

function __init__()
    storycloze_init()
end

include("./storycloze.jl")


end
