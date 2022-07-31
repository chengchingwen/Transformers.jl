using Transformers.Basic
using Transformers.HuggingFace
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

const _bert_model = hgf"bert-base-uncased:forsequenceclassification"
const bertenc = hgf"bert-base-uncased:tokenizer"

const bert_model = todevice(_bert_model)

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
                y = bert_model(data.input.tok, data.label; token_type_ids = data.input.segment)
                (y.loss, y.logits)
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
        p = bert_model(data.input.tok, data.label; token_type_ids = data.input.segment).logits
        a = acc(p, data.label)
        al += a
        i+=1
    end
    al /= i
    @info "testing" accuracy = al
    return al
end

