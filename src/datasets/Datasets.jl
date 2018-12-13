module Datasets

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

include("./wmt.jl")
using .WMT




end
