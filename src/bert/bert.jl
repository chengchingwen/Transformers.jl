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
"""
function Bert(size::Int, head::Int, ps::Int, layer::Int;
              act = gelu, pdrop = 0.1, attn_pdrop = 0.1)
  rem(size,  head) != 0 && error("size not divisible by head")
  Bert(size, head, div(size, head), ps, layer; act=act, pdrop=pdrop, attn_pdrop=attn_pdrop)
end

function Bert(size::Int, head::Int, hs::Int, ps::Int, layer::Int; act = gelu, pdrop = 0.1, attn_pdrop = 0.1)
  Bert(
    Stack(
      @nntopo_str("x':x => $layer"),
      [
        Transformer(size, head, hs, ps; future=true, act=act, pdrop=attn_pdrop)
        for i = 1:layer
      ]...
    ),
    Dropout(pdrop))
end

function (bert::Bert)(x::T, mask=nothing; all::Bool=false) where T
  e = bert.drop(x)
  t, ts = bert.ts(e)
  t = mask === nothing ? t : t .* mask
  if all
    t, ts
  else
    t
  end
end

# function masklmloss(embed::Embed{T}, transform, output_bias, et, t::AbstractArray{T, N}, posis::AbstractArray{Int, 2}, mask) where {T,N}
#   masktok = gather(t, posis)
#   s = size(masktok)
#   sim = logsoftmax(transpose(embed.embedding) * reshape(transform(masktok), s[1], :) .+ output_bias)




# end


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
