using DelimitedFiles

cola_init() = register(DataDep(
    "GLUE-CoLA",
    """
    The Corpus of Linguistic Acceptability (CoLA) task (GLUE version)
    """,
    "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4",
    "f212fcd832b8f7b435fb991f101abf89f96b933ab400603bf198960dfc32cbff";
    post_fetch_method = fn -> begin
      mv(fn, "CoLA.zip")
      DataDeps.unpack("CoLA.zip")
      innerdir = "CoLA"
      innerfiles = readdir(innerdir)
      mv.(joinpath.(innerdir, innerfiles), innerfiles)
      rm(innerdir)
    end
))

struct CoLA <: Dataset end

function trainfile(::CoLA)
    sets = readdlm(datadep"GLUE-CoLA/train.tsv", '\t', String)
    [selectdim(sets, 2, i) for i = (4,2)]
end

function devfile(::CoLA)
    sets = readdlm(datadep"GLUE-CoLA/dev.tsv", '\t', String)
    [selectdim(sets, 2, i) for i = (4,2)]
end

function testfile(::CoLA)
    sets, headers = readdlm(datadep"GLUE-CoLA/test.tsv", '\t', String; header=true)
    [selectdim(sets, 2, 2)]
end

get_labels(::CoLA) = ("0", "1")
