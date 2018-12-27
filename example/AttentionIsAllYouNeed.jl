"""
Reference: The Annotated Transformer (http://nlp.seas.harvard.edu/2018/04/03/attention.html)
"""

using Flux
using Flux: onehotbatch, onecold, onehot, crossentropy
using Flux.Tracker: back!

using Transformers
using Transformers.Basic: PositionEmbedding, NNTopo
using Transformers.Datasets: WMT, Train

using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--gpu", "-g"
            help = "use gpu"
            action = :store_true
        "task"
            help = "task name"
            required = true
            range_tester = x-> x ∈ ["wmt14", "copy"]
    end

    return parse_args(s)
end

args = parse_commandline()

use_gpu(args["gpu"])

if args["task"] == "copy"
    const N = 2
    const V = 10
    const Smooth = 1e-6

    startsym = 11
    endsym = 12
    unksym = 0
    labels = [unksym, startsym, endsym, collect(1:V)...]
    embedding = device(param(randn(512, length(labels))))

    function gen_data()
        global V
        d = rand(1:V, 10)
        (d,d)
    end

    function train!()
        println("start training")
        i = 1
        for i = 1:10000
            l = loss(gen_data())
            back!(l)
            i%500 == 0 && (@show l; opt())
        end
    end

    mkline(x) = [startsym, x..., endsym]
elseif args["task"] == "wmt14"
    const N = 6
    const Smooth = 0.4

    wmt14 = WMT.GoogleWMT()

    datas = dataset(Train, wmt14)
    vocab = get_vocab(wmt14)

    startsym = "<s>"
    endsym = "</s>"
    unksym = "</unk>"
    labels = [unksym, startsym, endsym, collect(keys(vocab))...]
    embedding = device(param(randn(512, length(labels))))

    function train!()
        println("start training")
        i = 1
        while (batch = get_batch(datas)) != []
            l = loss(batch[1])
            back!(l)
            i+=1
            i%64 == 0 && (@show l; opt())
        end
    end

    mkline(x) = [startsym, split(x)..., endsym]
else
    error("task not define")
end

embed(x) = (embedding * x) ./ sqrt(512)

encoder = device(Stack(
    NNTopo("e → pe:(e, pe) → x → $N"),
    PositionEmbedding(512),
    (e, pe) -> (e .+ pe),
    [Transformer(512, 8, 64, 2048) for i = 1:N]...
))

decoder = device(Stack(
    NNTopo("(e, m, mask):e → pe:(e, pe) → (t:(t, m, mask) → t:(t, m, mask)) → $N:t → c"),
    PositionEmbedding(512),
    (e, pe) -> (e .+ pe),
    [TransformerDecoder(512, 8, 64, 2048) for i = 1:N]...,
    Chain(Dense(512, length(labels)), logsoftmax)
))

opt = ADAM(params(embedding, encoder, decoder), 1e-4; β2=0.98)

kl_div(q, logp) = sum(q .* (log.(q .+ eps(q[1])) .- logp)) / size(q, 2)

loss((x,t)) = loss(x, t)
function loss(x, t)
    global Smooth
    ix = [startsym, mkline(x)..., endsym]
    iy = [startsym, mkline(t)..., endsym]
    ex = onehotbatch(ix, labels, unksym)
    et = onehotbatch(iy, labels, unksym)
    src = embed(ex)
    trg = embed(et)

    enc = encoder(src)
    dec = decoder(trg, enc, nothing)

    #label smoothing
    label = device((fill(Smooth, size(et)) .+ et .* (1 - 2*Smooth))[:, 2:end])

    loss = kl_div(label, dec[:, 1:end-1])
end

function translate(x)
    ex = onehotbatch([startsym, mkline(x)..., endsym], labels, unksym)
    et = onehotbatch([startsym], labels, unksym)

    len = size(ex)[end]

    src = embed(ex)
    trg = embed(et)


    enc = encoder(src)

    seq = [startsym]

    for i = 1:2len
        trg = embed(onehotbatch(seq, labels, unksym))
        dec = decoder(trg, enc, nothing)
        @show ntok = onecold(dec, labels)
        push!(seq, ntok[end])
        ntok[end] == endsym && break
    end
    seq
end
