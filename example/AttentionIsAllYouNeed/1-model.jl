"""
Reference: The Annotated Transformer (http://nlp.seas.harvard.edu/2018/04/03/attention.html)
"""

using Flux
using Flux: onecold, gradient
import Flux.Optimise: update!

using WordTokenizers

using Transformers
using Transformers.Basic

using Random
Random.seed!(0)

include("./0-data.jl")

const vocab = Vocabulary(labels, unksym)
const embed = todevice(Embed(512, length(vocab); scale=inv(sqrt(512))))

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
    lab = onehot(vocab, trg)

    src = m.embed(src)
    trg = m.embed(trg)

    if src_mask === nothing || trg_mask === nothing
        mask = nothing
    else
        mask = getmask(src_mask, trg_mask)
    end

    enc = m.encoder(src)
    dec = m.decoder(trg, enc, mask)

    #label smoothing
    label = smooth(lab)[:, 2:end, :]

    if mask == nothing
        loss = logkldivergence(label, dec[:, 1:end-1, :])
    else
        loss = logkldivergence(label, dec[:, 1:end-1, :], trg_mask[:, 1:end-1, :])
    end
end

function translate(x)
    ix = todevice(vocab(mkline(x)))
    seq = [startsym]

    src = embed(ix)
    enc = encoder(src)

    len = length(ix)
    for i = 1:2len
        trg = embed(todevice(vocab(seq)))
        dec = decoder(trg, enc, nothing)
        #move back to gpu due to argmax wrong result on CuArrays
        ntok = onecold(collect(dec), labels)
        push!(seq, ntok[end])
        ntok[end] == endsym && break
    end
    seq
end


# train!()
