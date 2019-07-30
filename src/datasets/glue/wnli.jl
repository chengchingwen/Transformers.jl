import ..Datasets: dataset, datafile, reader, Mode, Test

wnli_init() = register(DataDep(
    "GLUE-WNLI",
    """
    Winograd NLI (WNLI) task (GLUE version)
    """,
    "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FWNLI.zip?alt=media&token=068ad0a0-ded7-4bd7-99a5-5e00222e0faf",
    "ae0e8e4d16f4d46d4a0a566ec7ecceccfd3fbfaa4a7a4b4e02848c0f2561ac46";
    post_fetch_method = fn -> begin
      mv(fn, "WNLI.zip")
      DataDeps.unpack("WNLI.zip")
      innerdir = "WNLI"
      innerfiles = readdir(innerdir)
      mv.(joinpath.(innerdir, innerfiles), innerfiles)
      rm(innerdir)
    end
))

struct WNLI <: Dataset end

function dataset(::Type{M}, d::WNLI) where M <: Mode
    ds = reader(datafile(M, d))
    header = split(take!(ds), '\t')
    field_num = length(header)
    needed_field = M == Test ? (2,3) : (2,3,4)
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

function trainfile(::WNLI)
    datadep"GLUE-WNLI/train.tsv"
end

function devfile(::WNLI)
    datadep"GLUE-WNLI/dev.tsv"
end

function testfile(::WNLI)
    datadep"GLUE-WNLI/test.tsv"
end

get_labels(::WNLI) = ("0", "1")

