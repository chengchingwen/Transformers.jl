"""
Reference: The Annotated Transformer (http://nlp.seas.harvard.edu/2018/04/03/attention.html)
"""

using ArgParse

using Flux
using Flux: onecold
using Flux.Tracker: back!

using Transformers
using Transformers.Basic: PositionEmbedding, Embed, getmask, onehot, NNTopo
using Transformers.Datasets: WMT, Train, batched


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
    const Batch = 16

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
        for i = 1:300
            data = batched([gen_data() for i = 1:Batch])
            @time l = loss(data)
            @time back!(l)
            i%8 == 0 && (@show l; @time opt())
        end
    end

    mkline(x) = [startsym, x..., endsym]
elseif args["task"] == "wmt14"
    const N = 6
    const Smooth = 0.4
    const Batch = 16

    wmt14 = WMT.GoogleWMT()

    datas = dataset(Train, wmt14)
    vocab = get_vocab(wmt14)

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
            i%5 == 0 && (@show l; @time opt())
        end
    end

    mkline(x) = [startsym, split(x)..., endsym]
else
    error("task not define")
end


#extend for 3d op
function (d::Dense)(x::AbstractArray{T, 3}) where T
    s = size(x)
    reshape(d(reshape(x, s[1], :)), size(d.W, 1), s[2], s[3])
end

logsoftmax3d(x) = logsoftmax(x)
function logsoftmax3d(x::AbstractArray{T, 3}) where T
    s = size(x)
    reshape(logsoftmax(reshape(x, s[1], :)), s)
end

embed = Embed(512, labels, unksym)

function embedding(x)
    em, ma = embed(x)
    #sqrt(512) makes type unstable
    em ./ convert(typeof(em.data[1]),sqrt(512)), ma
end

broadcast_add(e, pe) = e .+ pe
function broadcast_add(e::AbstractArray{T, 3}, pe) where T
    #for Flux gpu issue 530 https://github.com/FluxML/Flux.jl/issues/530
    s = size(e)
    reshape(reshape(e, :, s[end]) .+ reshape(pe, :, 1), s)
end

encoder = device(Stack(
    NNTopo("e → pe:(e, pe) → x → $N"),
    PositionEmbedding(512),
    broadcast_add,
    # (e, pe) -> (e .+ pe),
    [Transformer(512, 8, 64, 2048) for i = 1:N]...
))

decoder = device(Stack(
    NNTopo("(e, m, mask):e → pe:(e, pe) → (t:(t, m, mask) → t:(t, m, mask)) → $N:t → c"),
    PositionEmbedding(512),
    broadcast_add,
    # (e, pe) -> (e .+ pe),
    [TransformerDecoder(512, 8, 64, 2048) for i = 1:N]...,
    Chain(Dense(512, length(labels)), logsoftmax3d)
))


opt = ADAM(params(embed, encoder, decoder), 1e-4; β2=0.98)

kl_div(q::AbstractArray{T, 3},
       logp::AbstractArray{T, 3},
       mask) where T =
           kl_div(reshape(q, size(q, 1), :), reshape(logp, size(logp, 1), :), reshape(mask, 1, :))

function kl_div(q, logp, mask)
    kld = (q .* (log.(q .+ eps(q[1])) .- logp)) #handle gpu broadcast error
    sum(kld .* mask) / sum(mask)
end

loss((x,t)) = loss(x, t)
function loss(x, t)
    global Smooth
    ix = mkline.(x)
    iy = mkline.(t)
    et = onehot(embed, iy)

    src, src_mask = embedding(ix)
    trg, trg_mask = embedding(iy)

    mask = getmask(src_mask, trg_mask)

    enc = encoder(src)
    dec = decoder(trg, enc, mask)

    #label smoothing
    label = device((fill(Smooth/length(embed.vocab), size(et)) .* (1 .- et) .+ et .* (1 - Smooth))[:, 2:end, :])

    loss = kl_div(label, dec[:, 1:end-1, :], trg_mask[:, 1:end-1, :])
end

function translate(x)
    ix = mkline(x)
    seq = [startsym]

    src, _ = embedding(ix)
    trg, _ = embedding(seq)
    enc = encoder(src)

    len = length(ix)
    for i = 1:2len
        trg, _ = embedding(seq)
        dec = decoder(trg, enc, nothing)
        @show ntok = onecold(dec, labels)
        push!(seq, ntok[end])
        ntok[end] == endsym && break
    end
    seq
end
