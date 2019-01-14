"""
Reference: The origin code of GPT paper from openai (https://github.com/openai/finetune-transformer-lm)
"""

using ArgParse

using Flux
using Flux: onecold, onehotbatch
using Flux.Tracker: back!

using BytePairEncoding

using Transformers
using Transformers.Basic: onehot, logcrossentropy
using Transformers.GenerativePreTrain
using Transformers.Datasets: StoryCloze, Train, Test, batched


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--gpu", "-g"
            help = "use gpu"
            action = :store_true
        "task"
            help = "task name"
            required = true
            range_tester = x -> x ∈ ["rocstories"]
    end

    return parse_args(s)
end

const args = parse_commandline()

use_gpu(args["gpu"])

const startsym = "_start_"
const delisym = "_deli_"
const clfsym = "_clf_"
const unksym = "<unk>"
const anslabel = ("1", "2")
gptm, embedm, bpe = load_gpt_pretrain(12;
                                    startsym=startsym,
                                    delisym=delisym,
                                    clfsym=clfsym,
                                    unksym=unksym)

const gpt = device(gptm)
const embed = device(embedm)
const clf = device(Dense(768, 1))

function transform(s1, s2, s3, s4, c1, c2, y)
    x = [startsym;
         segment(bpe, s1);
         segment(bpe, s2);
         segment(bpe, s3);
         segment(bpe, s4);
         delisym]
    x1 = [x; segment(bpe, c1); clfsym]
    x2 = [x; segment(bpe, c2); clfsym]

    x1, x2, y
end

function acc(p, y)
    pred = onecold(p, anslabel)
    sum(pred .== y) / length(y)
end

function loss(x1, x2, y)
    e1, e1_mask = embed(x1)
    e2, e2_mask = embed(x2)
    t1 = gpt(e1)
    t2 = gpt(e2)
    lm = lmloss(embed, onehot(embed, x1), t1, e1_mask) + lmloss(embed, onehot(embed, x2), t2, e2_mask)
    c1 = hcat(map(enumerate(findfirst(isequal(clfsym), x) for x in x1)) do (i, ind)
              t1[:, ind, i]
              end...
              )
    c2 = hcat(map(enumerate(findfirst(isequal(clfsym), x) for x in x2)) do (i, ind)
              t2[:, ind, i]
              end...
              )
    p1 = clf(c1)
    p2 = clf(c2)
    p = vcat(p1, p2)
    cl = logcrossentropy(device(onehotbatch(y, anslabel)), p)

    0.5lm + cl, p
end


const rocs = StoryCloze()
const datas = dataset(Train, rocs)
const opt = ADAM(params(embed, gpt, clf), 1e-4; β2=0.98)



const Batch = 4
function train!()
    global Batch
    println("start training")
    i::Int = 1
    a::Float64 = 0.
    while (batch = get_batch(datas, Batch)) != []
        tdb = transform.(batch...)
        b1, b2, y = batched(tdb)
        @time l, p = loss(b1, b2, y)
        a += acc(p, y)
        @time back!(l)
        i+=1
        i%5 == 0 && (@show a/5; a = 0.; @time opt())
    end
end
