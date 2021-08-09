import ..Datasets: dataset, datafile, reader, Mode, Test

qnli_init() = register(DataDep(
    "GLUE-QNLI",
    """
    Question NLI (SQuAD2.0 / QNLI) task (GLUE version)
    """,
    "https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip",
    "e634e78627a29adaecd4f955359b22bf5e70f2cbd93b493f2d624138a0c0e5f5";
    post_fetch_method = fn -> begin
      mv(fn, "QNLIv2.zip")
      DataDeps.unpack("QNLIv2.zip")
      innerdir = "QNLI"
      innerfiles = readdir(innerdir)
      mv.(joinpath.(innerdir, innerfiles), innerfiles)
      rm(innerdir)
    end
))

struct QNLI <: Dataset end

function dataset(::Type{M}, d::QNLI) where M <: Mode
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

function trainfile(::QNLI)
    datadep"GLUE-QNLI/train.tsv"
end

function devfile(::QNLI)
    datadep"GLUE-QNLI/dev.tsv"
end

function testfile(::QNLI)
    datadep"GLUE-QNLI/test.tsv"
end

get_labels(::QNLI) = ("entailment", "not_entailment")
