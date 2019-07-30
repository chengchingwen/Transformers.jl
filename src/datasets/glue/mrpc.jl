using DelimitedFiles

mrpc_init() = register(DataDep(
    "GLUE-MRPC",
    """
    The Microsoft Research Paraphrase Corpus (MRPC) task (GLUE version)
    """,
    [
        [
            "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt",
            "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt"
        ],
        "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc",
     ],
    "57fae0e7ccce8cd20caae9590117d5ced83a3529637b99ed12a8465bf17895ca";
    post_fetch_method = [
        identity,
        fn -> begin
          mv(fn, "mrpc_dev_ids.tsv")

          dev_ids = open("mrpc_dev_ids.tsv") do fn
            map(eachline(fn)) do line
              ids = Tuple(parse.(Int, split(strip(line), '\t')))
            end
          end

          trainset = open("msr_paraphrase_train.txt") do fn
            header = readline(fn)
            map(eachline(fn)) do line
              label, id1, id2, s1, s2 = split(strip(line), '\t')
            end
          end

          devset = filter(((label, id1, id2, s1, s2),)->parse.(Int, (id1, id2)) in dev_ids, trainset)
          trainset = filter!(!(((label, id1, id2, s1, s2),)->parse.(Int, (id1, id2)) in dev_ids), trainset)

          open("train.tsv", "w+") do tfn
            writedlm(tfn, trainset)
          end

          open("dev.tsv", "w+") do dfn
            writedlm(dfn, devset)
          end
          rm("msr_paraphrase_train.txt")
          rm("mrpc_dev_ids.tsv")

          testset = open("msr_paraphrase_test.txt") do fn
            header = readline(fn)
            map(eachline(fn)) do line
              label, id1, id2, s1, s2 = split(strip(line), '\t')
            end
          end
          open("test.tsv", "w+") do tfn
            writedlm(tfn, testset)
          end
          rm("msr_paraphrase_test.txt")
        end
    ]
))

struct MRPC <: Dataset end

function trainfile(::MRPC)
    sets = readdlm(datadep"GLUE-MRPC/train.tsv", '\t', String)
    [selectdim(sets, 2, i) for i = (4,5,1)]
end

function devfile(::MRPC)
    sets = readdlm(datadep"GLUE-MRPC/dev.tsv", '\t', String)
    [selectdim(sets, 2, i) for i = (4,5,1)]
end

function testfile(::MRPC)
    sets = readdlm(datadep"GLUE-MRPC/test.tsv", '\t', String)
    [selectdim(sets, 2, i) for i = (4,5,1)]
end

get_labels(::MRPC) = ("0", "1")
