using Flux: @treelike
using MacroTools: @forward

using ..Basic
using ..Basic: AbstractTransformer
using ..Stacks

struct Bert <: AbstractTransformer
  ts::Stack
  drop::Dropout
end

@treelike Bert

@forward Bert.ts Base.getindex, Base.length

"""
    Bert(size::Int, head::Int, ps::Int, layer::Int;
        act = gelu, pdrop = 0.1, attn_pdrop = 0.1)
    Bert(size::Int, head::Int, hs::Int, ps::Int, layer::Int;
        act = gelu, pdrop = 0.1, attn_pdrop = 0.1)

the Bidirectional Encoder Representations from Transformer(BERT) model.


    (bert::Bert)(x, mask=nothing; all::Bool=false)

eval the bert layer on input `x`. If length `mask` is given (in shape (1, seq_len, batch_size)), mask the attention with `getmask(mask, mask)`. Moreover, set `all` to `true` to get all 
outputs of each transformer layer.
"""
function Bert(size::Int, head::Int, ps::Int, layer::Int;
              act = gelu, pdrop = 0.1, attn_pdrop = 0.1)
  rem(size,  head) != 0 && error("size not divisible by head")
  Bert(size, head, div(size, head), ps, layer; act=act, pdrop=pdrop, attn_pdrop=attn_pdrop)
end

function Bert(size::Int, head::Int, hs::Int, ps::Int, layer::Int; act = gelu, pdrop = 0.1, attn_pdrop = 0.1)
  Bert(
    Stack(
      @nntopo_str("((x, m) => x':(x, m)) => $layer"),
      [
        Transformer(size, head, hs, ps; future=true, act=act, pdrop=attn_pdrop)
        for i = 1:layer
      ]...
    ),
    Dropout(pdrop))
end

function (bert::Bert)(x::T, mask=nothing; all::Bool=false) where T
  e = bert.drop(x)

  if mask === nothing
    t, ts = bert.ts(e, nothing)
  else
    t, ts = bert.ts(e, getmask(mask, mask))
  end

  if all
    if mask !== nothing
      ts = map(ts) do ti
        ti .* mask
      end
    end
    ts[end], ts
  else
    t = mask === nothing ? t : t .* mask
    t
  end
end

"""
    masklmloss(embed::Embed{T}, transform,
               t::AbstractArray{T, N}, posis::AbstractArray{Tuple{Int,Int}}, labels) where {T,N}
    masklmloss(embed::Embed{T}, transform, output_bias,
               t::AbstractArray{T, N}, posis::AbstractArray{Tuple{Int,Int}}, labels) where {T,N}

helper function for computing the maks language modeling loss. 
Performance `transform(x) .+ output_bias` where `x` is the mask specified by 
`posis`, then compute the similarity with `embed.embedding` and crossentropy between true `labels`.
"""
function masklmloss(embed::Embed{T}, transform, t::AbstractArray{T, N}, posis::AbstractArray{Tuple{Int,Int}}, labels) where {T,N}
  masktok = gather(t, posis)
  sim = logsoftmax(transpose(embed.embedding) * transform(masktok))
  return logcrossentropy(labels, sim)
end

function masklmloss(embed::Embed{T}, transform, output_bias, t::AbstractArray{T, N}, posis::AbstractArray{Tuple{Int,Int}}, labels) where {T,N}
  masktok = gather(t, posis)
  sim = logsoftmax(transpose(embed.embedding) * transform(masktok) .+ output_bias)
  return logcrossentropy(labels, sim)
end


function Base.show(io::IO, bert::Bert)
  hs = div(size(bert.ts[1].mh.iqproj.W)[1], bert.ts[1].mh.head)
  h, ps = size(bert.ts[1].pw.dout.W)

  print(io, "Bert(")
  print(io, "layers=$(length(bert.ts)), ")
  print(io, "head=$(bert.ts[1].mh.head), ")
  print(io, "head_size=$(hs), ")
  print(io, "pwffn_size=$(ps), ")
  print(io, "size=$(h))")
end
