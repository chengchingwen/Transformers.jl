using Flux
using Flux: gradient, onehot
import Flux.Optimise: update!

using WordTokenizers
using TextEncodeBase

using Transformers
using Transformers.Basic
using Transformers.Datasets
using Transformers.Datasets: IWSLT

const N = 6
const Smooth = 0.4
const Epoch = 1
const Batch = 8
const lr = 1e-6
const MaxLen = 100

const iwslt2016 = IWSLT.IWSLT2016(:en, :de)
const word_counts = get_vocab(iwslt2016)

const startsym = "<s>"
const endsym = "</s>"
const unksym = "</unk>"
const labels = [unksym; startsym; endsym; collect(keys(word_counts))]

const textenc = Basic.TransformerTextEncoder(labels; startsym, endsym, unksym,
                                             padsym = unksym, trunc = MaxLen)

function preprocess(batch)
    global textenc
    x, x_mask = encode(textenc, batch[1])
    t, t_mask = encode(textenc, batch[2])
    todevice(x,t,x_mask,t_mask)
end

function train!()
    global Epoch, Batch
    println("start training")
    model = (embed=embed, encoder=encoder, decoder=decoder)
    i = 1
    for e = 1:Epoch
        datas = dataset(Train, iwslt2016)
        while (batch = get_batch(datas, Batch)) |> !isempty
            x, t, x_mask, t_mask = preprocess(batch)
            grad = gradient(ps) do
                loss(model, x, t, x_mask, t_mask)
            end
            i+=1
            i%8 == 0 && @show loss(model, x, t, x_mask, t_mask)
            update!(opt, ps, grad)
        end
    end
    return model
end

const embed = todevice(Embed(512, length(textenc.vocab); scale=inv(sqrt(512))))

const encoder = todevice(Stack(
    @nntopo(e → pe:(e, pe) → x → x → $N),
    PositionEmbedding(512),
    .+,
    Dropout(0.1),
    [Transformer(512, 8, 64, 2048) for i = 1:N]...
))

const decoder = todevice(Stack(
    @nntopo((e, m, mask):e → pe:(e, pe) → t → (t:(t, m, mask) → t:(t, m, mask)) → $N:t → c),
    PositionEmbedding(512),
    .+,
    Dropout(0.1),
    [TransformerDecoder(512, 8, 64, 2048) for i = 1:N]...,
    Positionwise(Dense(512, length(textenc.vocab)), logsoftmax)
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
    label = @view smooth(trg)[:, 2:end, :]

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

function translate(x::AbstractString)
    ix = todevice(encode(textenc, x).tok)
    seq = [startsym]

    src = embed(ix)
    enc = encoder(src)

    len = size(ix, 2)
    for i = 1:2len
        trg = embed(todevice(lookup(textenc, seq)))
        dec = decoder(trg, enc, nothing)
        ntok = decode(textenc, argmax(@view(dec[:, end])))
        push!(seq, ntok)
        ntok == endsym && break
    end
    seq
end
