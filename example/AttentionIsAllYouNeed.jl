"""
Reference: The Annotated Transformer (http://nlp.seas.harvard.edu/2018/04/03/attention.html)
"""

using ArgParse

using Flux
using Flux: onecold, gradient
import Flux.Optimise: update!

using WordTokenizers

using Transformers
using Transformers.Basic
using Transformers.Datasets
using Transformers.Datasets: WMT, IWSLT


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

const args = parse_commandline()

if args["gpu"]
    @eval using CuArrays
end

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
        for i = 1:320*7
            data = batched([gen_data() for i = 1:Batch])
            x, t = data
            x = mkline.(x)
            t = mkline.(t)
            x_mask = getmask(x)
            t_mask = getmask(t)
            x, t = embed.Vocab(x, t)
            x, t, x_mask, t_mask = todevice(x,t,x_mask,t_mask)
            l = loss(x, t, x_mask, t_mask)
            grad = gradient(()->l, ps)
            i%8 == 0 && @show l
            update!(opt, ps, grad)
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
            x = mkline.(batch[1])
            t = mkline.(batch[2])
            x_mask = getmask(x)
            t_mask = getmask(t)
            x, t = embed.Vocab(x, t)
            x, t, x_mask, t_mask = todevice(x,t,x_mask,t_mask)
            l = loss(x,t, x_mask, t_mask)
            grad = gradient(()->l, ps)
            i+=1
            #i%5 == 0 &&
            (@show l; @time update!(opt, ps, grad))
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

vocab = Vocabulary(labels, unksym)
const embed = gpu(Embed(512, vocab))

embedding(x) = embed(x, inv(sqrt(512)))


const encoder = gpu(Stack(
    NNTopo("e → pe:(e, pe) → x → x → $N"),
    PositionEmbedding(512),
    (e, pe) -> e .+ pe,
    Dropout(0.1),
    [Transformer(512, 8, 64, 2048) for i = 1:N]...
))

const decoder = gpu(Stack(
    NNTopo("(e, m, mask):e → pe:(e, pe) → t → (t:(t, m, mask) → t:(t, m, mask)) → $N:t → c"),
    PositionEmbedding(512),
    (e, pe) -> e .+ pe,
    Dropout(0.1),
    [TransformerDecoder(512, 8, 64, 2048) for i = 1:N]...,
    Sequence(Dense(512, length(labels)), logsoftmax)
))

const ps = params(embed, encoder, decoder)
const opt = ADAM(lr)

function smooth(et)
    global Smooth
    sm = fill!(similar(et, Float32), Smooth/length(embed.Vocab))
    p = sm .* (1 .+ -et)
    label = p .+ et .* (1 - convert(Float32, Smooth))
    label
end

function loss(src, trg, src_mask, trg_mask)
    lab = onehot(embed, trg)

    src = embedding(src)
    trg = embedding(trg)

    if src_mask === nothing || trg_mask === nothing
        mask = nothing
    else
        mask = getmask(src_mask, trg_mask)
    end

    enc = encoder(src)
    dec = decoder(trg, enc, mask)

    #label smoothing
    label = smooth(lab)[:, 2:end, :]

    loss = logkldivergence(label, dec[:, 1:end-1, :], trg_mask[:, 1:end-1, :])
end

function translate(x)
    ix = todevice(embed.Vocab(mkline(x))
    seq = [startsym]

    src = embedding(ix)
    enc = encoder(src)

    len = length(ix)
    for i = 1:2len
        trg = embedding(todevice(embed.Vocab(seq)))
        dec = decoder(trg, enc, nothing)
        #move back to gpu due to argmax wrong result on CuArrays
        ntok = onecold(collect(dec), labels)
        push!(seq, ntok[end])
        ntok[end] == endsym && break
    end
    seq
end


train!()
