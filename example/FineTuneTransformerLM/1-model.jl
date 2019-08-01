"""
Reference: The origin code of GPT paper from openai (https://github.com/openai/finetune-transformer-lm)
"""

using Flux
using Flux: onehotbatch, onecold, gradient, logitcrossentropy
import Flux.Optimise: update!

using BytePairEncoding

using Transformers
using Transformers.Basic
using Transformers.GenerativePreTrain
using Transformers.Pretrain

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

include("./0-data.jl")

const startsym = "_start_"
const delisym = "_deli_"
const clfsym = "_clf_"
const unksym = "<unk>"
const labels = ("1", "2")

gpt_model, bpe, vocab, tokenizer = load_pretrain("GPT-OpenAIftlm";
                                                 startsym=startsym,
                                                 delisym=delisym,
                                                 clfsym=clfsym,
                                                 unksym=unksym)


set_tokenizer(tokenizer)

const gpt = gpu(Basic.set_classifier(gpt_model, Chain(Dense(768, 1), Dropout(0.1))))
const embed = gpt.embed.embeddings.tok

function acc(p, y)
    pred = onecold(collect(p))
    label = onecold(collect(y))
    sum(pred .== label) / length(label)
end

function loss(x1, x2, y, x1_mask, x2_mask, c1_index, c2_index)
    e1 = gpt.embed(tok=x1)
    e2 = gpt.embed(tok=x2)
    t1 = gpt.transformers(e1, x1_mask)
    t2 = gpt.transformers(e2, x2_mask)
    lm = lmloss(embed, onehot(vocab, x1), t1, x1_mask) + lmloss(embed, onehot(vocab, x2), t2, x2_mask)

    c1 = gather(t1, c1_index)
    c2 = gather(t2, c2_index)

    p1 = gpt.classifier(c1)::typeof(c1)
    p2 = gpt.classifier(c2)::typeof(c1)
    p = vcat(p1, p2)::typeof(c1)

    cl = logitcrossentropy(p, y)::typeof(lm)
    #unstable type will cause performance issue
    convert(Float32, 0.5) * lm + cl, p
end

const rocs = StoryCloze()
const ps = params(gpt)
const opt = ADAM(6.25e-5)

train!(args["epoch"])
