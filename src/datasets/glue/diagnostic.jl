using Dates
using DelimitedFiles

using HTTP

import ..Datasets: dataset, datafile, reader, Mode, Test

diagnostic_init() = register(DataDep(
    "GLUE-Diagnostic",
    """
    GLUE Diagnostics Datasets
    """,
    "https://storage.googleapis.com/mtl-sentence-representations.appspot.com/tsvsWithoutLabels%2FAX.tsv?GoogleAccessId=firebase-adminsdk-0khhl@mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&Signature=DuQ2CSPt2Yfre0C%2BiISrVYrIFaZH1Lc7hBVZDD4ZyR7fZYOMNOUGpi8QxBmTNOrNPjR3z1cggo7WXFfrgECP6FBJSsURv8Ybrue8Ypt%2FTPxbuJ0Xc2FhDi%2BarnecCBFO77RSbfuz%2Bs95hRrYhTnByqu3U%2FYZPaj3tZt5QdfpH2IUROY8LiBXoXS46LE%2FgOQc%2FKN%2BA9SoscRDYsnxHfG0IjXGwHN%2Bf88q6hOmAxeNPx6moDulUF6XMUAaXCSFU%2BnRO2RDL9CapWxj%2BDl7syNyHhB7987hZ80B%2FwFkQ3MEs8auvt5XW1%2Bd4aCU7ytgM69r8JDCwibfhZxpaa4gd50QXQ%3D%3D",
    "0e13510b1bb14436ff7e2ee82338f0efb0133ecf2e73507a697dc210db3f05fd";
    fetch_method=(url, localdir)-> begin
      format_progress(x) = round(x, digits=4)
      format_bytes(x) = !isfinite(x) ? "∞ B" : Base.format_bytes(x)
      format_seconds(x) = "$(round(x; digits=2)) s"
      format_bytes_per_second(x) = format_bytes(x) * "/s"

      filepath = joinpath(localdir, "diagnostic.tsv")

      HTTP.open("GET", url) do stream
        resp = HTTP.startread(stream)

        total_bytes = parse(Float64, HTTP.header(resp, "Content-Length"))
        downloaded_bytes = 0
        start_time = now()
        prev_time = now()

        function report_callback()
            prev_time = now()
            taken_time = (prev_time - start_time).value / 1000 # in seconds
            average_speed = downloaded_bytes / taken_time
            remaining_bytes = total_bytes - downloaded_bytes
            remaining_time = remaining_bytes / average_speed
            completion_progress = downloaded_bytes / total_bytes

            @info("Downloading",
                  source=url,
                  dest = filepath,
                  progress = completion_progress |> format_progress,
                  time_taken = taken_time |> format_seconds,
                  time_remaining = remaining_time |> format_seconds,
                  average_speed = average_speed |> format_bytes_per_second,
                  downloaded = downloaded_bytes |> format_bytes,
                  remaining = remaining_bytes |> format_bytes,
                  total = total_bytes |> format_bytes,
                  )
        end


        Base.open(filepath, "w") do fh
            while(!eof(stream))
                downloaded_bytes += write(fh, readavailable(stream))
                if now() - prev_time > Millisecond(1000*2)
                    report_callback()
                end
            end
        end
        report_callback()
      end
      filepath
    end
))

struct Diagnostic <: Dataset end

function testfile(::Diagnostic)
    sets, header = readlm(datadep"GLUE-Diagnostic/diagnostic.tsv", '\t', String; header=true)
    [selectdim(sets, 2, i) for i = (2,3)]
end
