using Flux: @treelike
using MacroTools: @forward

using ..Basic
using ..Basic: onehot, AbstractTransformer
using ..Stacks

struct Gpt <: AbstractTransformer
    ts::Stack
    drop::Dropout
end

@treelike Gpt

@forward Gpt.ts Base.getindex, Base.length

"""
    Gpt(size::Int, head::Int, ps::Int, layer::Int;
        act = gelu, pdrop = 0.1, attn_pdrop = 0.1)
    Gpt(size::Int, head::Int, hs::Int, ps::Int, layer::Int;
        act = gelu, pdrop = 0.1, attn_pdrop = 0.1)

the Generative Pretrained Transformer(GPT) model.

    (gpt::Gpt)(x::T, mask=nothing; all::Bool=false)

eval the gpt layer on input `x`. If length `mask` is given (in shape (1, seq_len, batch_size)), mask the attention with `mask`. Moreover, set `all` to `true` to get all 
outputs of each transformer layer.
"""
function Gpt(size::Int, head::Int, ps::Int, layer::Int;
             act = gelu, pdrop = 0.1, attn_pdrop = 0.1)
    rem(size, head) != 0 && error("size not divisible by head")
    Gpt(size, head, div(size, head), ps, layer; act=act, pdrop=pdrop, attn_pdrop=attn_pdrop)
end

function Gpt(size::Int, head::Int, hs::Int, ps::Int, layer::Int;
             act = gelu, pdrop = 0.1, attn_pdrop = 0.1)
    Gpt(
        Stack(
            @nntopo_str("x':x => $layer"),
            [
                Transformer(size, head, hs, ps; future=false, act=act, pdrop=attn_pdrop)
                for i = 1:layer
            ]...
        ),
        Dropout(pdrop)
    )
end

function (gpt::Gpt)(x::T, mask=nothing; all::Bool=false) where T
    e = gpt.drop(x)
    t, ts = gpt.ts(e)
    t = mask === nothing ? t : t .* mask
    if all
        t, ts
    else
        t #size(t) == (size, seq_len, batch)
    end
end


"""
    lmloss(embed, onehot, encoding, mask)

compute the language modeling loss for Gpt, onehot is the onehot array of the origin
input sentence. encoding the output of Gpt, mask is the mask between input sentences.

"""
lmloss(embed::Embed{T}, o::OneHotArray, t::AbstractArray{T}, mask) where T = lmloss(embed, tofloat(T, o), t, mask)
function lmloss(embed::Embed{T}, et, t::AbstractArray{T, N}, mask) where {T,N}
    if N == 3
        t = t[:, 1:end-1, :]
        s = size(t)
        sim = logsoftmax(transpose(embed.embedding) * reshape(t, s[1], :)) #(vocab, seq_len*batch)
        sim = reshape(sim, :, s[2], s[3])
        return logcrossentropy(et[:, 2:end, :], sim, mask[:, 2:end, :])
    elseif N == 2
        t = t[:, 1:end-1]
        sim = logsoftmax(transpose(embed.embedding) * t)
        return logcrossentropy(et[:, 2:end], sim, mask[:, 2:end])
    end
end

function Base.show(io::IO, gpt::Gpt)
    hs = div(size(gpt.ts[1].mh.iqproj.W)[1], gpt.ts[1].mh.head)
    h, ps = size(gpt.ts[1].pw.dout.W)

    print(io, "Gpt(")
    print(io, "layers=$(length(gpt.ts)), ")
    print(io, "head=$(gpt.ts[1].mh.head), ")
    print(io, "head_size=$(hs), ")
    print(io, "pwffn_size=$(ps), ")
    print(io, "size=$(h))")
end
