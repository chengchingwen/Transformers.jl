module Datasets

export Dataset, Train, Dev, Test
export dataset, datafile, get_batch, get_vocab

"drive.google.com"
const dgdst = joinpath(dirname(@__FILE__), "download_gd_to.sh")

function download_gdrive(url, localdir)
    cmd = `sh $dgdst "$url" "$localdir"`
    filepath = chomp(read(cmd, String))
    filename = basename(filepath)
    mv(joinpath(dirname(@__FILE__), filename), filepath)
    filepath
end


include("./dataset.jl")

include("translate/wmt.jl")
using .WMT

include("translate/iwslt.jl")
using .IWSLT


end
