using Transformers
using Transformers.TextEncoders
using Transformers.HuggingFace
using Transformers.Datasets
using Transformers.Datasets: GLUE

using Flux
using Flux.Losses
using Zygote
import Optimisers

const Epoch = 2
const Batch = 4
const mrpc = GLUE.MRPC()
const labels = Vocab([get_labels(mrpc)...])

function preprocess(batch)
    global labels
    data = encode(bertenc, map(collect, zip(batch[1], batch[2])))
    label = lookup(OneHot, labels, batch[3])
    return merge(data, (label = label,))
end

# load the old config file and update some value
const bert_config = HuggingFace.HGFConfig(hgf"bert-base-uncased:config"; num_labels = length(labels))

# load the model / tokenizer with new config
const _bert_model = load_model("bert-base-uncased", :ForSequenceClassification; config = bert_config)
const bertenc = load_tokenizer("bert-base-uncased"; config = bert_config)

const bert_model = todevice(_bert_model)

const opt_rule = Optimisers.Adam(1e-6)
const opt = Optimisers.setup(opt_rule, bert_model)

function acc(p, label)
    pred = Flux.onecold(p)
    truth = Flux.onecold(label)
    sum(pred .== truth) / length(truth)
end

function loss(model, input)
    nt = model(input)
    p = nt.logit
    l = logitcrossentropy(p, input.label)
    return l, p
end

function train!()
    global Batch, Epoch, bert_model
    @info "start training: $(args["task"])"
    for e = 1:Epoch
        @info "epoch: $e"
        datas = dataset(Train, mrpc)
        i = 1
        al = zero(Float64)
        while (batch = get_batch(datas, Batch)) !== nothing
            input = todevice(preprocess(batch))
            (l, p), back = Zygote.pullback(bert_model) do model
                loss(model, input)
            end
            a = acc(p, input.label)
            al += a
            (grad,) = back((Zygote.sensitivity(l), nothing))
            i += 1
            Optimisers.update!(opt, bert_model, grad)
            mod1(i, 16) == 1 && @info "training" loss=l accuracy=al/i
        end

        test()
    end
end

function test()
    @info "testing"
    i = 1
    al = zero(Float64)
    datas = dataset(Dev, mrpc)
    while (batch = get_batch(datas, Batch)) !== nothing
        input = todevice(preprocess(batch))
        p = bert_model(input).logit
        a = acc(p, input.label)
        al += a
        i+=1
    end
    al /= i
    @info "testing" accuracy = al
    return al
end
