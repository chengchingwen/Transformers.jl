using DelimitedFiles

cola_init() = register(DataDep(
    "GLUE-CoLA",
    """
    The Corpus of Linguistic Acceptability (CoLA) task (GLUE version)
    """,
    "https://dl.fbaipublicfiles.com/glue/data/CoLA.zip",
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
