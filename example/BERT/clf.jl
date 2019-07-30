using Transformers
using Transformers.Basic
using Transformers.Datasets
using Transformers.Datasets: GLUE
using Transformers.BidirectionalEncoder

using Flux
using Flux: onehotbatch, gradient
import Flux.Optimise: update!
using WordTokenizers

include("./args.jl")

const Epoch = 20
const Batch = 4

if args["task"] == "cola"
    const task = GLUE.CoLA()

    markline(sent) = ["[CLS]"; sent; "[SEP]"]
    function preprocess(batch)
        sentence = markline.(wordpiece.(tokenizer.(batch[1])))
        mask = getmask(sentence)
        tok = vocab(sentence)
        segment = fill!(similar(tok), 1)

        label = onehotbatch(batch[2], labels)
        return (tok=tok, segment=segment), label, mask
    end
else
    if args["task"] == "mnli"
        const task = GLUE.MNLI(false)
    elseif args["task"] == "mrpc"
        const task = GLUE.MRPC()
    end
    markline(s1, s2) = ["[CLS]"; s1; "[SEP]"; s2; "[SEP]"]
    function preprocess(batch)
        s1 = wordpiece.(tokenizer.(batch[1]))
        s2 = wordpiece.(tokenizer.(batch[2]))
        sentence = markline.(s1, s2)
        mask = getmask(sentence)
        tok = vocab(sentence)

        segment = fill!(similar(tok), 1)
        for (i, sent) ∈ enumerate(sentence)
            j = findfirst(isequal("[SEP]"), sent)
            if j !== nothing
                @view(segment[j+1:end, i]) .= 2
            end
        end

        label = onehotbatch(batch[3], labels)
        return (tok=tok, segment=segment), label, mask
    end
end

const labels = get_labels(task)

const _bert_model, wordpiece, tokenizer = load_bert_pretrain("../../src/bert/uncased_L-12_H-768_A-12.tfbson", :all)
const bert_model = gpu(_bert_model)
const vocab = Vocabulary(wordpiece)


const hidden_size = size(bert_model.classifier.pooler.W ,1)
const clf = gpu(Chain(
    Dense(hidden_size, length(labels)),
    logsoftmax
))

const ps = params(bert_model)
const opt = ADAM(1e-4)


function loss(data, label, mask=nothing)
    e = bert_model.embed(data)
    t = bert_model.transformers(e, mask)
    l = Basic.logcrossentropy(
        clf(
            bert_model.classifier.pooler(
                t[:,1,:]
            )
        ),
        label
    )
    return l
end

function train!()
    global Batch
    global Epoch
    @info "start training: $(args["task"])"
    for e = 1:Epoch
        @info "epoch: $e"
        datas = dataset(Train, task)

        i = 1
        while (batch = get_batch(datas, Batch)) !== nothing
            data, label, mask = todevice(preprocess(batch))
            @time l = loss(data, label, mask)
            @show l
            grad = gradient(()->l, ps)
            @time update!(opt, ps, grad)
        end
    end
end


