using Transformers.Basic
using Transformers.Pretrain
using Transformers.Datasets
using Transformers.Datasets: StoryCloze
using Transformers.GenerativePreTrain

using Flux
using Flux: pullback, params
import Flux.Optimise: update!

const Epoch = 2
const Batch = 4

const rocs = StoryCloze()

const startsym = "_start_"
const delisym = "_deli_"
const clfsym = "_clf_"
const unksym = "<unk>"

function preprocess(batch)
    global gptenc, labels
    data1 = encode(gptenc, batched(batch[[1:4; 5]]))
    data2 = encode(gptenc, batched(batch[[1:4; 6]]))
    label = lookup(OneHot, labels, batch[end])
    return (data1 = data1, data2 = data2, label = label)
end

const labels = Basic.Vocab([get_labels(rocs)...])

const _gpt_model, bpe, vocab, tokenizer = load_pretrain("GPT-OpenAIftlm"; startsym, delisym, clfsym, unksym)
const gptenc = GPTTextEncoder(tokenizer, bpe, vocab; startsym, sepsym = delisym, endsym = clfsym, unksym, padsym = unksym)

const gpt_model = todevice(
    set_classifier(_gpt_model, Chain(Dense(768, 1)))
)

const ps = params(gpt_model)
const opt = ADAM(6.25e-5)

function acc(p, label)
    pred = Flux.onecold(p)
    truth = Flux.onecold(label)
    sum(pred .== truth) / length(truth)
end

function loss(model, data)
    e1 = model.embed(data.data1.input)
    e2 = model.embed(data.data2.input)
    t1 = model.transformers(e1, data.data1.mask)
    t2 = model.transformers(e2, data.data2.mask)
    lm1 = lmloss(model.embed.embeddings.tok, data.data1.input.tok, t1, data.data1.mask)
    lm2 = lmloss(model.embed.embeddings.tok, data.data2.input.tok, t2, data.data2.mask)
    lm = lm1 + lm2
    p1 = model.classifier(t1[:, end, :])
    p2 = model.classifier(t2[:, end, :])
    p = logsoftmax(vcat(p1, p2))
    cl = Basic.logcrossentropy(data.label, p)
    return 0.5f0 * lm + cl, p
end

function train!()
    global Batch
    global Epoch
    @info "start training: $(args["task"])"
    for e = 1:Epoch
        @info "epoch: $e"
        Flux.trainmode!(gpt_model)
        datas = dataset(Train, rocs)

        i = 1
        al = zero(Float64)
        while (batch = get_batch(datas, Batch)) !== nothing
            data = todevice(preprocess(batch::Vector{Vector{String}}))
            (l, p), back = pullback(ps) do
                loss(gpt_model, data)
            end
            a = acc(p, data.label)
            al += a
            grad = back((Flux.Zygote.sensitivity(l), nothing))
            i+=1
            update!(opt, ps, grad)
            mod1(i, 16) == 1 && @info "training" loss=l accuracy=al/i
        end

        test()
    end
end

function test()
    @info "testing"
    Flux.testmode!(gpt_model)
    i = 1
    al = zero(Float64)
    datas = dataset(Test, rocs)
    while (batch = get_batch(datas, Batch)) !== nothing
      data = todevice(preprocess(batch::Vector{Vector{String}}))
      _, p = loss(gpt_model, data)
      a = acc(p, data.label)
      al += a
      i+=1
    end
    al /= i
    @info "testing" accuracy = al
    return al
end
