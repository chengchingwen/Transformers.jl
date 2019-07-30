import ..Datasets: dataset, datafile, reader, Mode, Test

rte_init() = register(DataDep(
    "GLUE-RTE",
    """
    Recognizing Textual Entailment (RTE) task (GLUE version)
    """,
    "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FRTE.zip?alt=media&token=5efa7e85-a0bb-4f19-8ea2-9e1840f077fb",
    "6bf86de103ecd335f3441bd43574d23fef87ecc695977a63b82d5efb206556ee";
    post_fetch_method = fn -> begin
      mv(fn, "RTE.zip")
      DataDeps.unpack("RTE.zip")
      innerdir = "RTE"
      innerfiles = readdir(innerdir)
      mv.(joinpath.(innerdir, innerfiles), innerfiles)
      rm(innerdir)
    end
))

struct RTE <: Dataset end

function dataset(::Type{M}, d::RTE) where M <: Mode
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

function trainfile(::RTE)
    datadep"GLUE-RTE/train.tsv"
end

function devfile(::RTE)
    datadep"GLUE-RTE/dev.tsv"
end

function testfile(::RTE)
    datadep"GLUE-RTE/test.tsv"
end

get_labels(::RTE) = ("entailment", "not_entailment")
