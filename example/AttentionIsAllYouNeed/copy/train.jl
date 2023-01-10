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

# configuration
const N = 2
const V = 10
const Smooth = 1e-6
const Batch = 32
const lr = 1e-4

# text encoder / preprocess
const startsym = "11"
const endsym = "12"
const unksym = "0"
const labels = [unksym, startsym, endsym, collect(map(string, 1:V))...]

const textenc = TransformerTextEncoder(split, labels; startsym, endsym, unksym, padsym = unksym)

function gen_data()
    global V
    d = join(rand(1:V, 10), ' ')
    (d,d)
end

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

# model definition
const hidden_dim = 512
const head_num = 8
const head_dim = 64
const ffn_dim = 2048

const token_embed = todevice(Embed(hidden_dim, length(textenc.vocab); scale = inv(sqrt(hidden_dim))))
const embed = Layers.CompositeEmbedding(token = token_embed, pos = SinCosPositionEmbed(hidden_dim))
const embed_decode = EmbedDecoder(token_embed)
const encoder = todevice(Transformer(TransformerBlock       , N, head_num, hidden_dim, head_dim, ffn_dim))
const decoder = todevice(Transformer(TransformerDecoderBlock, N, head_num, hidden_dim, head_dim, ffn_dim))
const seq2seq = Seq2Seq(encoder, decoder)
const trf_model = Layers.Chain(
    Layers.Parallel{(:encoder_input, :decoder_input)}(
        Layers.Chain(embed, todevice(Dropout(0.1)))),
    seq2seq,
    Layers.Branch{(:logits,)}(embed_decode),
)

const ps = params(trf_model)
const opt = ADAM(lr)

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

function train!()
    global Batch, trf_model
    println("start training")
    for i = 1:320*7
        data = batched([gen_data() for i = 1:Batch])
        input = preprocess(data)
        grad = gradient(ps) do
            nt = trf_model(input)
            shift_decode_loss(nt.logits, input.decoder_input.token, input.decoder_input.attention_mask)
        end
        i%8 == 0 && @show shift_decode_loss(trf_model(input).logits,
                                            input.decoder_input.token, input.decoder_input.attention_mask)
        update!(opt, ps, grad)
    end
end

function translate(x::AbstractString)
    ix = todevice(encode(textenc, x).token)
    seq = [startsym]

    src = embed((token = ix,))
    enc = encoder(src).hidden_state

    len = size(ix, 2)
    for i = 1:2len
        trg = embed((token = todevice(lookup(textenc, seq)),))
        dec = embed_decode(decoder(merge(trg, (memory = enc,))).hidden_state)
        ntok = decode(textenc, argmax(@view(dec[:, end])))
        push!(seq, ntok)
        ntok == endsym && break
    end
    seq
end
