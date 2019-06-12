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
        act = gelu, pdrop = 0.1, att_pdrop = 0.1)
    Bert(size::Int, head::Int, hs::Int, ps::Int, layer::Int;
        act = gelu, pdrop = 0.1, ayy_pdrop = 0.1)

the Bidirectional Encoder Representations from Transformer(BERT) model.
"""
function Bert(size::Int, head::Int, ps::Int, layer::Int;
              act = gelu, pdrop = 0.1, att_pdrop = 0.1)
  rem(size,  head) != 0 && error("size not divisible by head")
  Bert(size, head, div(size, head), ps, layer; act=act, pdrop=pdrop, att_pdrop=att_pdrop)
end

function Bert(size::Int, head::Int, hs::Int, ps::Int, layer::Int; act = gelu, pdrop = 0.1, att_pdrop = 0.1)
  Bert(
    Stack(
      @nntopo_str("x':x => $layer"),
      [
        Transformer(size, head, hs, ps; future=true, act=act, pdrop=att_pdrop)
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
