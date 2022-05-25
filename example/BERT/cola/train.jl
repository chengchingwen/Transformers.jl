using Transformers.Basic
using Transformers.Pretrain
using Transformers.Datasets
using Transformers.Datasets: GLUE
using Transformers.BidirectionalEncoder

using Flux
using Flux: pullback, params
import Flux.Optimise: update!
using WordTokenizers

const Epoch = 2
const Batch = 4

const cola = GLUE.CoLA()

function preprocess(batch)
    global bertenc, labels
    data = encode(bertenc, batch[1])
    label = lookup(OneHot, labels, batch[2])
    return merge(data, (label = label,))
end

const labels = Basic.Vocab([get_labels(cola)...])

const _bert_model, wordpiece, tokenizer = pretrain"Bert-uncased_L-12_H-768_A-12"
const bertenc = BertTextEncoder(tokenizer, wordpiece)

const hidden_size = size(_bert_model.classifier.pooler.weight, 1)
const clf = todevice(Chain(
    Dropout(0.1),
    Dense(hidden_size, length(labels)),
    logsoftmax
))

const bert_model = todevice(
    set_classifier(_bert_model,
                   (
                       pooler = _bert_model.classifier.pooler,
                       clf = clf
                   )
                  )
)

const ps = params(bert_model)
const opt = ADAM(1e-6)

function acc(p, label)
    pred = Flux.onecold(p)
    truth = Flux.onecold(label)
    sum(pred .== truth) / length(truth)
end

function loss(model, data)
    e = model.embed(data.input)
    t = model.transformers(e, data.mask)

    p = model.classifier.clf(
        model.classifier.pooler(
            t[:,1,:]
        )
    )

    l = Basic.logcrossentropy(data.label, p)
    return l, p
end

function train!()
    global Batch
    global Epoch
    @info "start training: $(args["task"])"
    for e = 1:Epoch
        @info "epoch: $e"
        Flux.trainmode!(bert_model)
        datas = dataset(Train, cola)

        i = 1
        al = zero(Float64)
        while (batch = get_batch(datas, Batch)) !== nothing
            data = todevice(preprocess(batch))
            (l, p), back = pullback(ps) do
                loss(bert_model, data)
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
    Flux.testmode!(bert_model)
    i = 1
    al = zero(Float64)
    datas = dataset(Dev, cola)
    while (batch = get_batch(datas, Batch)) !== nothing
      data = todevice(preprocess(batch))
      _, p = loss(bert_model, data)
      a = acc(p, data.label)
      al += a
      i+=1
    end
    al /= i
    @info "testing" accuracy = al
    return al
end

