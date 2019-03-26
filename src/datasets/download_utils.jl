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

function find_gcode(ckj)
    for cookie ∈ ckj
        if match(r"_warning_", cookie.name) !== nothing
            return cookie.value
        end
    end

    nothing
end


function download_gdrive(url, localdir)
    rq = HTTP.request("GET", url; cookies=true)
    ckj = HTTP.CookieRequest.default_cookiejar["drive.google.com"]
    gcode = find_gcode(ckj)
    @assert gcode !== nothing

    rq = HTTP.request("GET", "$url&confirm=$gcode"; cookies=true)
    hcd = HTTP.header(rq, "Content-Disposition")
    m = match(r"filename=\\\"(.*)\\\"", hcd)
    if m === nothing
        filename = "gdrive_downloaded"
    else
        filename = m.captures[]
    end

    filepath = joinpath(localdir, filename)

    open(filepath, "w+") do f
        write(f, rq.body)
    end

    filepath
end
