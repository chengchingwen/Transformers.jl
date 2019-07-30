import ..Datasets: dataset, datafile, reader, Mode, Test

qqp_init() = register(DataDep(
    "GLUE-QQP",
    """
    Quora Question Pairs (QQP) task (GLUE version)
    """,
    "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQQP.zip?alt=media&token=700c6acf-160d-4d89-81d1-de4191d02cb5",
    "67cb8f5fe66c90a0bc1bf5792e3924f63008b064ab7a473736c919d20bb140ad";
    post_fetch_method = fn -> begin
      mv(fn, "QQP.zip")
      DataDeps.unpack("QQP.zip")
      innerdir = "QQP"
      innerfiles = readdir(innerdir)
      mv.(joinpath.(innerdir, innerfiles), innerfiles)
      rm(innerdir)
    end
))

struct QQP <: Dataset end

function dataset(::Type{M}, d::QQP) where M <: Mode
    ds = reader(datafile(M, d))
    header = split(take!(ds), '\t')
    field_num = length(header)
    needed_field = M == Test ? (2,3) : (4,5,6)
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

function trainfile(::QQP)
    datadep"GLUE-QQP/train.tsv"
end

function devfile(::QQP)
    datadep"GLUE-QQP/dev.tsv"
end

function testfile(::QQP)
    datadep"GLUE-QQP/test.tsv"
end

get_labels(::QQP) = ("0", "1")

