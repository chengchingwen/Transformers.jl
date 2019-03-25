"""
Reference: The Annotated Transformer (http://nlp.seas.harvard.edu/2018/04/03/attention.html)
"""

using ArgParse

using Flux
using Flux: onecold
using Flux.Tracker: back!
import Flux.Optimise: update!

using WordTokenizers

using Transformers
using Transformers.Basic: NNTopo
using Transformers.Basic: PositionEmbedding, Embed, getmask, onehot,
                          logkldivergence, Sequence
using Transformers.Datasets: WMT, IWSLT, Train, batched


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--gpu", "-g"
            help = "use gpu"
            action = :store_true
        "task"
            help = "task name"
            required = true
            range_tester = x-> x ∈ ["wmt14", "iwslt2016", "copy"]
    end

    return parse_args(ARGS, s)
end

args = parse_commandline()

use_gpu(args["gpu"])

if args["task"] == "copy"
    const N = 2
    const V = 10
    const Smooth = 1e-6
    const Batch = 32
    const lr = 1e-4

    startsym = 11
    endsym = 12
    unksym = 0
    labels = [unksym, startsym, endsym, collect(1:V)...]

    function gen_data()
        global V
        d = rand(1:V, 10)
        (d,d)
    end

    function train!()
        global Batch
        println("start training")
        i = 1
        for i = 1:320*15
            data = batched([gen_data() for i = 1:Batch])
            @time l = loss(data)
            @time back!(l)
            i%8 == 0 && (@show l; update!(opt, ps))
        end
    end

    mkline(x) = [startsym, x..., endsym]
elseif args["task"] == "wmt14" || args["task"] == "iwslt2016"
    const N = 6
    const Smooth = 0.4
    const Batch = 8
    const lr = 1e-6
    const MaxLen = 100

    const task = args["task"]

    if task == "wmt14"
        wmt14 = WMT.GoogleWMT()

        datas = dataset(Train, wmt14)
        vocab = get_vocab(wmt14)
    else
        iwslt2016 = IWSLT.IWSLT2016(:en, :de)

        datas = dataset(Train, iwslt2016)
        vocab = get_vocab(iwslt2016)
    end

    startsym = "<s>"
    endsym = "</s>"
    unksym = "</unk>"
    labels = [unksym, startsym, endsym, collect(keys(vocab))...]

    function train!()
        global Batch
        println("start training")
        i = 1
        while (batch = get_batch(datas, Batch)) != []
            @time l = loss(batch)
            @time back!(l)
            i+=1
            i%5 == 0 && (@show l; @time update!(opt, ps))
        end
    end

    if task == "wmt14"
        function mkline(x)
            global MaxLen
            xi = split(x)
            if length(xi) > MaxLen
                xi = xi[1:100]
            end

            [startsym, xi..., endsym]
        end
    else
        function mkline(x)
            global MaxLen
            xi = tokenize(x)
            if length(xi) > MaxLen
                xi = xi[1:100]
            end

            [startsym, xi..., endsym]
        end
    end
else
    error("task not define")
end

embed = device(Embed(512, labels, unksym))

function embedding(x)
    em, ma = embed(x)
    #sqrt(512) makes type unstable
    em ./ convert(get_ftype(), sqrt(512)), ma
end

encoder = device(Stack(
    NNTopo("e → pe:(e, pe) → x → x → $N"),
    PositionEmbedding(512),
    (e, pe) -> e .+ pe,
    Dropout(0.1),
    [Transformer(512, 8, 64, 2048) for i = 1:N]...
))

decoder = device(Stack(
    NNTopo("(e, m, mask):e → pe:(e, pe) → t → (t:(t, m, mask) → t:(t, m, mask)) → $N:t → c"),
    PositionEmbedding(512),
    (e, pe) -> e .+ pe,
    Dropout(0.1),
    [TransformerDecoder(512, 8, 64, 2048) for i = 1:N]...,
    Sequence(Dense(get_ftype(), 512, length(labels)), logsoftmax)
))

ps = params(embed, encoder, decoder)
opt = ADAM(lr)

function smooth(et)
    global Smooth
    sm = device(fill(Smooth/length(embed.vocab), size(et)))
    p = sm .* (1 .+ -et)
    label = p .+ et .* (1 - convert(get_ftype(), Smooth))
    label
end

loss((x,t)) = loss(x, t)
function loss(x, t)
    ix = mkline.(x)
    iy = mkline.(t)
    et = onehot(embed, iy)

    src, src_mask = embedding(ix)
    trg, trg_mask = embedding(iy)

    mask = getmask(src_mask, trg_mask)

    enc = encoder(src)
    dec = decoder(trg, enc, mask)

    #label smoothing
    label = smooth(et)[:, 2:end, :]

    loss = logkldivergence(label, dec[:, 1:end-1, :], trg_mask[:, 1:end-1, :])
end

function translate(x)
    ix = mkline(x)
    seq = [startsym]

    src, _ = embedding(ix)
    enc = encoder(src)

    len = length(ix)
    for i = 1:2len
        trg, _ = embedding(seq)
        dec = decoder(trg, enc, nothing)
        ntok = onecold(dec, labels)
        push!(seq, ntok[end])
        ntok[end] == endsym && break
    end
    seq
end


train!()
