import ..Datasets: dataset, datafile, reader, Mode, Test

mnli_init() = register(DataDep(
    "GLUE-MNLI",
    """
     The Multi-Genre Natural Language Inference (MNLI) task (GLUE version)
    """,
    "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce",
    "e7c1d896d26ed6caf700110645df426cc2d8ebf02a5ab743d5a5c68ac1c83633";
    post_fetch_method = fn -> begin
      mv(fn, "MNLI.zip")
      DataDeps.unpack("MNLI.zip")
      innerdir = "MNLI"
      innerfiles = readdir(innerdir)
      mv.(joinpath.(innerdir, innerfiles), innerfiles)
      rm(innerdir)
    end
))

struct MNLI <: Dataset
    matched::Bool
end

function dataset(::Type{M}, d::MNLI) where M <: Mode
    ds = reader(datafile(M, d))
    header = split(take!(ds), '\t')
    field_num = length(header)
    needed_field = (9,10,12)
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

function trainfile(::MNLI)
    datadep"GLUE-MNLI/train.tsv"
end

function devfile(mnli::MNLI)
    if mnli.matched
        datadep"GLUE-MNLI/dev_matched.tsv"
    else
        datadep"GLUE-MNLI/dev_mismatched.tsv"
    end
end

function testfile(::MNLI)
    if mnli.matched
        datadep"GLUE-MNLI/test_matched.tsv"
    else
        datadep"GLUE-MNLI/test_mismatched.tsv"
    end
end

get_labels(::MNLI) = ("entailment", "neutral", "contradiction")
