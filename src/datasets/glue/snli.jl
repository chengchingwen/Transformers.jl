import ..Datasets: dataset, datafile, reader, Mode, Test

snli_init() = register(DataDep(
    "GLUE-SNLI",
    """
    The Stanford Natural Language Inference (SNLI) task (GLUE version)
    """,
    "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSNLI.zip?alt=media&token=4afcfbb2-ff0c-4b2d-a09a-dbf07926f4df",
    "48c972c3d3590cb79227cd91fda7319ac14068ce804e703364524e171b53dc16";
    post_fetch_method = fn -> begin
      mv(fn, "SNLI.zip")
      DataDeps.unpack("SNLI.zip")
      innerdir = "SNLI"
      innerfiles = readdir(innerdir)
      mv.(joinpath.(innerdir, innerfiles), innerfiles)
      rm(innerdir)
    end
))

struct SNLI <: Dataset end

function dataset(::Type{M}, d::SNLI) where M <: Mode
    ds = reader(datafile(M, d))
    header = split(take!(ds), '\t')
    field_num = length(header)
    needed_field = (8,9,11)
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

function trainfile(::SNLI)
    datadep"GLUE-SNLI/train.tsv"
end

function devfile(::SNLI)
    datadep"GLUE-SNLI/dev.tsv"
end

function testfile(::SNLI)
    datadep"GLUE-SNLI/test.tsv"
end

get_labels(::SNLI) = ("entailment", "neutral", "contradiction")
