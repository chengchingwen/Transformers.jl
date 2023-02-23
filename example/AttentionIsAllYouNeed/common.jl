using Statistics
using Flux
using Flux.Losses
import Optimisers
using Zygote
using ChainRulesCore

using Transformers
using Transformers.Layers
using Transformers.TextEncoders
using Transformers.Datasets

function preprocess(data)
    global textenc
    x, t = data
    input = encode(textenc, x, t)
    return todevice(input)
end

function smooth(et)
    global Smooth
    sm = fill!(similar(et, Float32), Smooth/length(textenc.vocab))
    p = sm .* (1 .+ -et)
    label = p .+ et .* (1 - convert(Float32, Smooth))
    return label
end
ChainRulesCore.@non_differentiable smooth(et)

function shift_decode_loss(logits, trg, trg_mask)
    label = @view smooth(trg)[:, 2:end, :]
    return logitcrossentropy(mean, @view(logits[:, 1:end-1, :]), label, trg_mask - 1)
end

function translate(x::AbstractString)
    global textenc, embed, encoder, decoder, embed_decode, startsym, endsym
    ix = todevice(encode(textenc, x).token)
    seq = [startsym]

    encoder_input = (token = ix,)
    src = embed(encoder_input)
    enc = encoder(src).hidden_state

    len = size(ix, 2)
    for i = 1:2len
        decoder_input = (token = todevice(lookup(textenc, seq)), memory = enc)
        trg = embed(decoder_input)
        dec = decoder(trg).hidden_state
        logit = embed_decode(dec)
        ntok = decode(textenc, argmax(@view(logit[:, end])))
        push!(seq, ntok)
        ntok == endsym && break
    end
    seq
end
