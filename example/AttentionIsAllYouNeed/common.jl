using Statistics
using Flux
using Flux: gradient, onehot, params
using Flux.Losses
import Flux.Optimise: update!
using ChainRulesCore

using Transformers
using Transformers.Layers
using Transformers.TextEncoders
using Transformers.Datasets

import Transformers.NeuralAttentionlib as NAlib


function preprocess(data)
    global textenc
    x, t = data
    x_data = encode(textenc, x)
    t_data = encode(textenc, t)
    input = (encoder_input = x_data,
             decoder_input = merge(t_data,
                                   (cross_attention_mask = NAlib.AttenMask(
                                       t_data.attention_mask, x_data.attention_mask),)))
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
