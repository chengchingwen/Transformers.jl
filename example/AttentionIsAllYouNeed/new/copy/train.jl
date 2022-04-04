using Flux
using Flux: onecold, gradient, onehot
import Flux.Optimise: update!

using WordTokenizers
using TextEncodeBase

using Transformers
using Transformers.Basic
using Transformers.Datasets

const N = 2
const V = 10
const Smooth = 1e-6
const Batch = 32
const lr = 1e-4

const startsym = "11"
const endsym = "12"
const unksym = "0"
const labels = [unksym, startsym, endsym, collect(map(string, 1:V))...]

const textenc = Basic.TransformerTextEncoder(labels; startsym, endsym, unksym, padsym = unksym)

function gen_data()
    global V
    d = join(rand(1:V, 10), ' ')
    (d,d)
end

function preprocess(data)
    global textenc
    x, t = data
    x, x_mask = encode(textenc, x)
    t, t_mask = encode(textenc, t)
    todevice(x,t,x_mask,t_mask)
end

function train!()
    global Batch
    println("start training")
    model = (embed=embed, encoder=encoder, decoder=decoder)
    i = 1
    for i = 1:320*7
        data = batched([gen_data() for i = 1:Batch])
        x, t, x_mask, t_mask = preprocess(data)
        grad = gradient(ps) do
            l = loss(model, x, t, x_mask, t_mask)
            l
        end
        i%8 == 0 && @show loss(model, x, t, x_mask, t_mask)
        update!(opt, ps, grad)
    end
end

const embed = todevice(Embed(512, length(textenc.vocab); scale=inv(sqrt(512))))

const encoder = todevice(Stack(
    @nntopo(e → pe:(e, pe) → x → x → $N),
    PositionEmbedding(512),
    (e, pe) -> e .+ pe,
    Dropout(0.1),
    [Transformer(512, 8, 64, 2048) for i = 1:N]...
))

const decoder = todevice(Stack(
    @nntopo((e, m, mask):e → pe:(e, pe) → t → (t:(t, m, mask) → t:(t, m, mask)) → $N:t → c),
    PositionEmbedding(512),
    (e, pe) -> e .+ pe,
    Dropout(0.1),
    [TransformerDecoder(512, 8, 64, 2048) for i = 1:N]...,
    Positionwise(Dense(512, length(labels)), logsoftmax)
))

const ps = params(embed, encoder, decoder)
const opt = ADAM(lr)

function smooth(et)
    global Smooth
    sm = fill!(similar(et, Float32), Smooth/size(embed, 2))
    p = sm .* (1 .+ -et)
    label = p .+ et .* (1 - convert(Float32, Smooth))
    label
end
Flux.@nograd smooth

function loss(m, src, trg, src_mask, trg_mask)
    #label smoothing
    label = smooth(trg)[:, 2:end, :]

    if isnothing(src_mask) || isnothing(trg_mask)
        mask = nothing
    else
        mask = getmask(src_mask, trg_mask)
    end

    enc = m.encoder(m.embed(src))
    dec = m.decoder(m.embed(trg), enc, mask)

    if isnothing(mask)
        loss = logkldivergence(label, dec[:, 1:end-1, :])
    else
        loss = logkldivergence(label, dec[:, 1:end-1, :], trg_mask[:, 1:end-1, :])
    end
end

function translate(x)
    ix = todevice(encode(textenc, x).tok)
    seq = [startsym]

    src = embed(ix)
    enc = encoder(src)

    len = size(ix, 2)
    for i = 1:2len
        trg = reshape(embed(todevice(lookup(textenc, seq))), Val(3))
        dec = decoder(trg, enc, nothing)
        ntok = onecold(dec, labels)
        push!(seq, ntok[end])
        ntok[end] == endsym && break
    end
    seq
end
