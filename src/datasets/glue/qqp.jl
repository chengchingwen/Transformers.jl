import ..Datasets: dataset, datafile, reader, Mode, Test

qqp_init() = register(DataDep(
    "GLUE-QQP",
    """
    Quora Question Pairs (QQP) task (GLUE version)
    """,
    "https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip",
    "40e7c862c04eb26ee04b67fd900e76c45c6ba8e6d8fab4f8f1f8072a1a3fbae0";
    post_fetch_method = fn -> begin
      mv(fn, "QQP-clean.zip")
      DataDeps.unpack("QQP-clean.zip")
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

