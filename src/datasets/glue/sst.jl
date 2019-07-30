using DelimitedFiles

sst_init() = register(DataDep(
    "GLUE-SST",
    """
    The Stanford Sentiment Treebank (SST) task (GLUE version)
    """,
    "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8",
    "d67e16fb55739c1b32cdce9877596db1c127dc322d93c082281f64057c16deaa";
    post_fetch_method = fn -> begin
      mv(fn, "SST-2.zip")
      DataDeps.unpack("SST-2.zip")
      innerdir = "SST-2"
      innerfiles = readdir(innerdir)
      mv.(joinpath.(innerdir, innerfiles), innerfiles)
      rm(innerdir)
    end
))

struct SST <: Dataset end

function trainfile(::SST)
    sets, headers = readdlm(datadep"GLUE-SST/train.tsv", '\t', String; header=true)
    [selectdim(sets, 2, i) for i = 1:2]
end

function devfile(::SST)
    sets, headers = readdlm(datadep"GLUE-SST/dev.tsv", '\t', String; header=true)
    [selectdim(sets, 2, i) for i = 1:2]
end

function testfile(::SST)
    sets, headers = readdlm(datadep"GLUE-SST/test.tsv", '\t', String; header=true)
    [selectdim(sets, 2, 2)]
end

get_labels(::SST) = ("0", "1")
