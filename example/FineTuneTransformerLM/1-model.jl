"""
Reference: The origin code of GPT paper from openai (https://github.com/openai/finetune-transformer-lm)
"""

using Flux
using Flux: onecold, gradient, logitcrossentropy
import Flux.Optimise: update!

using BytePairEncoding

using Transformers
using Transformers.Basic
using Transformers.GenerativePreTrain

include("./0-data.jl")

const startsym = "_start_"
const delisym = "_deli_"
const clfsym = "_clf_"
const unksym = "<unk>"
const anslabel = ["1", "2"]
const anv = Vocabulary(anslabel, "1")
gptm, embedm, bpe, vocab = load_gpt_pretrain(12;
                                             startsym=startsym,
                                             delisym=delisym,
                                             clfsym=clfsym,
                                             unksym=unksym)

const gpt = gpu(gptm)
const embed = gpu(embedm)
const clf = gpu(Dense(768, 1))

const ansdrop = Dropout(0.1)

function acc(p, y)
    pred = onecold(collect(p))
    sum(pred .== collect(y)) / length(y)
end

function loss(x1, x2, y, x1_mask, x2_mask, c1_index, c2_index)
    e1 = embed(x1)
    e2 = embed(x2)
    t1 = gpt(e1, x1_mask)
    t2 = gpt(e2, x2_mask)
    lm = lmloss(embed, onehot(vocab, x1), t1, x1_mask) + lmloss(embed, onehot(vocab, x2), t2, x2_mask)

    c1 = gather(t1, c1_index)
    c2 = gather(t2, c2_index)

    p1 = clf(c1)
    p2 = clf(c2)
    p = vcat(p1, p2)
    p = ansdrop(p, 1)

    ##### turn onehot to real float array
    yd = tofloat(Float32, onehot(anv, y))
    #####

    cl = logitcrossentropy(p, yd)
    #unstable type will cause performance issue
    convert(Float32, 0.5) * lm + cl, p
end

const rocs = StoryCloze()
const ps = params(embed, gpt, clf)
const opt = ADAM(6.25e-5)

train!(args["epoch"])
