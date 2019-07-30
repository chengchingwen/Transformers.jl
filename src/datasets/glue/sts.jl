import ..Datasets: dataset, datafile, reader, Mode, Test

sts_init() = register(DataDep(
    "GLUE-STS",
    """
     The Semantic Textual Similarity Benchmark (STS) task (GLUE version)
    """,
    "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSTS-B.zip?alt=media&token=bddb94a7-8706-4e0d-a694-1109e12273b5",
    "e60a6393de5a8b5b9bac5020a1554b54e3691f9d600b775bd131e613ac179c85";
    post_fetch_method = fn -> begin
      mv(fn, "STS-B.zip")
      DataDeps.unpack("STS-B.zip")
      innerdir = "STS-B"
      innerfiles = readdir(innerdir)
      mv.(joinpath.(innerdir, innerfiles), innerfiles)
      rm(innerdir)
    end
))

struct STS <: Dataset end

function dataset(::Type{M}, d::STS) where M <: Mode
    ds = reader(datafile(M, d))
    header = split(take!(ds), '\t')
    field_num = length(header)
    needed_field = (8,9,10)
    rds = get_channels(String, length(needed_field))

    task = @async begin
        for s ∈ ds
            s = split(s, '\t')
            length(s) != field_num && continue
            for (i, j) ∈ enumerate(needed_field)
                put!(rds[i], s[j])
            end
        end
    end
    for rd ∈ rds
        bind(rd, task)
    end

    rds
end

function trainfile(::STS)
    datadep"GLUE-STS/train.tsv"
end

function devfile(::STS)
    datadep"GLUE-STS/dev.tsv"
end

function testfile(::STS)
    datadep"GLUE-STS/test.tsv"
end
