"""
Reference: The origin code of GPT paper from openai (https://github.com/openai/finetune-transformer-lm)
"""

using ArgParse
using CuArrays

using Flux
using Flux: onecold, gradient, logitcrossentropy
import Flux.Optimise: update!

using BytePairEncoding

using Transformers
using Transformers.Basic
using Transformers.GenerativePreTrain
using Transformers.Datasets
using Transformers.Datasets: StoryCloze


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

const startsym = "_start_"
const delisym = "_deli_"
const clfsym = "_clf_"
const unksym = "<unk>"
const anslabel = ["1", "2"]
const anv = Vocabulary(anslabel, "1")
gptm, embedm, bpe = load_gpt_pretrain(12;
                                    startsym=startsym,
                                    delisym=delisym,
                                    clfsym=clfsym,
                                    unksym=unksym)

const gpt = gpu(gptm)
const embed = gpu(embedm)
const clf = gpu(Dense(768, 1))

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
    pred = onecold(collect(p))
    sum(pred .== collect(y)) / length(y)
end

function loss(x1, x2, y, x1_mask, x2_mask, c1_index, c2_index)
    e1 = embed(x1)
    e2 = embed(x2)
    t1 = gpt(e1, x1_mask)
    t2 = gpt(e2, x2_mask)
    lm = lmloss(embed, onehot(embed, x1), t1, x1_mask) + lmloss(embed, onehot(embed, x2), t2, x2_mask)
    c1 = gather(reshape(t1, size(t1,1), :), c1_index)
    c2 = gather(reshape(t2, size(t2,1), :), c2_index)
    # c1 = hcat(map(enumerate(findfirst(isequal(clfsym), x) for x in x1)) do (i, ind)
    #           t1[:, ind, i]
    #           end...)
    # c2 = hcat(map(enumerate(findfirst(isequal(clfsym), x) for x in x2)) do (i, ind)
    #           t2[:, ind, i]
    #           end...)

    drop = Dropout(0.1)
    p1 = clf(c1)
    p2 = clf(c2)
    p = vcat(p1, p2)
    p = drop(p, 1)

    ##### handle data placement
    yd = tofloat(Float32, onehot(anv, y))
    # oy = onehotarray(y, anslabel)
    # yd = copyto!(similar(p), oy)
    #####

    cl = logitcrossentropy(p, yd)
    #unstable type will cause performance issue
    convert(Float32, 0.5) * lm + cl, p
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
        b1_mask = getmask(b1)
        b2_mask = getmask(b2)
        b1, b2 = embed.Vocab.((b1,b2))
        c1i = findall(isequal(embed.Vocab(clfsym)), reshape(b1, :))
        c2i = findall(isequal(embed.Vocab(clfsym)), reshape(b2, :))
        y = anv(y)
        b1,b2,y,b1_mask,b2_mask,c1i,c2i = CuArray.((b1,b2,y,b1_mask,b2_mask,c1i,c2i))

        _, p = loss(b1, b2, y, b1_mask, b2_mask, c1i, c2i)
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
            b1_mask = getmask(b1)
            b2_mask = getmask(b2)
            b1, b2 = embed.Vocab.((b1,b2))
            c1i = findall(isequal(embed.Vocab(clfsym)), reshape(b1, :))
            c2i = findall(isequal(embed.Vocab(clfsym)), reshape(b2, :))
            y = anv(y)
            b1,b2,y,b1_mask,b2_mask,c1i,c2i = CuArray.((b1,b2,y,b1_mask,b2_mask,c1i,c2i))

            l, p = loss(b1, b2, y, b1_mask, b2_mask, c1i, c2i)
            #@show l
            a = acc(p, y)
            al += a
            grad = gradient(()->l, ps)
            i+=1
            update!(opt, ps, grad)
            i%16==0 && @show al/i
        end
        test()
    end
end

train!(args["epoch"])
