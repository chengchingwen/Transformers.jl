"""
Reference: The origin code of GPT paper from openai (https://github.com/openai/finetune-transformer-lm)
"""

using ArgParse

using Flux
using Flux: onecold, onehotbatch, logitcrossentropy
using Flux.Tracker: back!
import Flux.Optimise: update!

using BytePairEncoding

using Transformers
using Transformers.Basic: onehot, crossentropy
using Transformers.GenerativePreTrain
using Transformers.Datasets: StoryCloze, Train, Test, batched


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--gpu", "-g"
            help = "use gpu"
            action = :store_true
        "--epoch", "-e"
            help = "epoch"
            arg_type = Int
            default = 3
        "task"
            help = "task name"
            required = true
            range_tester = x -> x ∈ ["rocstories"]
    end

    return parse_args(ARGS, s)
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
    pred = onecold(collect(p), anslabel)
    sum(pred .== y) / length(y)
end

function loss(x1, x2, y)
    e1, e1_mask = embed(x1)
    e2, e2_mask = embed(x2)
    t1 = gpt(e1, e1_mask)
    t2 = gpt(e2, e2_mask)
    lm = lmloss(embed, onehot(embed, x1), t1, e1_mask) + lmloss(embed, onehot(embed, x2), t2, e2_mask)
    c1 = hcat(map(enumerate(findfirst(isequal(clfsym), x) for x in x1)) do (i, ind)
              t1[:, ind, i]
              end...)
    c2 = hcat(map(enumerate(findfirst(isequal(clfsym), x) for x in x2)) do (i, ind)
              t2[:, ind, i]
              end...)

    drop = Dropout(0.1)
    p1 = clf(c1)
    p2 = clf(c2)
    p = vcat(p1, p2)
    p = drop(p, 1)
    # cl = crossentropy(device(one(get_ftype()) * onehotbatch(y, anslabel)), p)
    cl = logitcrossentropy(p, device(one(get_ftype()) * onehotbatch(y, anslabel)))
    #unstable type will cause performance issue
    convert(get_ftype(), 0.5) * lm + cl, p
end

const rocs = StoryCloze()
const ps = params(embed, gpt, clf)
const opt = ADAM(6.25e-5)

const Batch = 4

function test()
    Flux.testmode!(gpt)
    println("eval:")
    i::Int = 0
    al::Float64 = 0.
    devl = dataset(Test, rocs)
    while (batch = get_batch(devl, Batch)) !== nothing
        tdb = transform.(batch...)
        b1, b2, y = batched(tdb)
        _, p = loss(b1, b2, y)
        a = acc(p, y)
        al += a
        i += 1
    end
    al /= i
    Flux.testmode!(gpt, false)
    @show al
end

function train!(epoch)
    global Batch, rocs, opt, ps
    for e = 1:epoch
        println("start training: $e")
        datas = dataset(Train, rocs)
        i::Int = 0
        al::Float64 = 0.
        while (batch = get_batch(datas, Batch)) !== nothing
            tdb = transform.(batch...)
            b1, b2, y = batched(tdb)
            l, p = loss(b1, b2, y)
            #@show l
            a = acc(p, y)
            al += a
            back!(l)
            i+=1
            i%8 == 0 && update!(opt, ps)
            i%16==0 && @show al/i
        end
        test()
    end
end

train!(args["epoch"])
