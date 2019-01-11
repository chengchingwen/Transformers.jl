"""
    unshortlink(url)

unshorten a short url with the service of http://checkshorturl.com,
 and return the url if it is not a short link
"""
function unshortlink(url)
    cmd = `curl "http://checkshorturl.com/expand.php?u=$url"`
    @show cmd
    html = read(cmd, String)
    m = match(r".*<td .*Long URL<\/td>\s*<td .*><a href=\"([^ ]*)\"", html)
    m === nothing ? url : m.captures[1]
end

"drive.google.com"
const dgdst = joinpath(dirname(@__FILE__), "download_gd_to.sh")

"download from google drive"
function download_gdrive(url, localdir)
    cmd = `sh $dgdst "$url" "$localdir"`
    filepath = chomp(read(cmd, String))
    filename = basename(filepath)
    mv(joinpath(dirname(@__FILE__), filename), filepath)
    filepath
end

isgooglesheet(url) = occursin("docs.google.com/spreadsheets", url)
isgoogledrive(url) = occursin("drive.google.com", url)

function googlesheet_handler(url; format=:csv)
    link, expo = splitdir(url)
    if startswith(expo, "edit") || expo == ""
        url = link * "/export?format=$format"
    elseif startswith(expo, "export")
        url = replace(url, r"format=([a-zA-Z]*)(.*)"=>SubstitutionString("format=$format\\2"))
    end
    url
end

function mybegoogle_download(url, localdir)
    long_url = unshortlink(url)
    if isgooglesheet(long_url)
        long_url = googlesheet_handler(long_url)
    end

    if isgoogledrive(url)
        download_gdrive(long_url, localdir)
    else
        DataDeps.fetch_http(long_url, localdir)
    end
end
