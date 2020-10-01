using Flux
using Flux: @functor

abstract type AbstractTransformer end

struct PwFFN{Di<:Dense, Do<:Dense}
    din::Di
    dout::Do
end

@functor PwFFN


"just a wrapper for two dense layer."
PwFFN(size::Int, h::Int, act = relu) = PwFFN(
    Dense(size, h, act),
    Dense(h, size)
)

function (pw::PwFFN)(x::AbstractMatrix)
  # size(x) == (dims, seq_len)
  pw.dout(pw.din(x))
end

function (pw::PwFFN)(x::A) where {T, N, A<:AbstractArray{T, N}}
  new_x = reshape(x, size(x, 1), :)
  y = pw(new_x)
  return reshape(y, Base.setindex(size(x), size(y, 1), 1))
end

struct Transformer{MA<:MultiheadAttention, LA<:LayerNorm, P<:PwFFN, LP<:LayerNorm, DP<:Dropout} <: AbstractTransformer
    mh::MA
    mhn::LA
    pw::P
    pwn::LP
    drop::DP
end

@functor Transformer


"""
    Transformer(size::Int, head::Int, ps::Int;
                future::Bool = true, act = relu, pdrop = 0.1)
    Transformer(size::Int, head::Int, hs::Int, ps::Int;
                future::Bool = true, act = relu, pdrop = 0.1)

Transformer layer.

`size` is the input size. if `hs` is not specify, use `div(size, head)` as the hidden size of multi-head attention. 
`ps` is the hidden size & `act` is the activation function of the positionwise feedforward layer. 
When `future` is `false`, the k-th token can't see the j-th tokens where j > k. `pdrop` is the dropout rate.
"""
function Transformer(size::Int, head::Int, ps::Int; future::Bool = true, act = relu, pdrop = 0.1)
    rem(size, head) != 0 && error("size not divisible by head")
    Transformer(size, head, div(size, head), ps;future=future, act=act, pdrop=pdrop)
end

Transformer(size::Int, head::Int, hs::Int, ps::Int; future::Bool = true, act = relu, pdrop = 0.1) = Transformer(
    MultiheadAttention(head, size, hs, size; future=future, pdrop=pdrop),
    LayerNorm(size),
    PwFFN(size, ps, act),
    LayerNorm(size),
    Dropout(pdrop),
)

function (t::Transformer)(x::A, mask=nothing) where {T, N, A<:AbstractArray{T, N}}
    dropout = t.drop
    a = t.mh(x, x, x; mask=mask)
    a = dropout(a)
    res_a = x + a
    res_a = t.mhn(res_a)
    pwffn = t.pw(res_a)
    pwffn = dropout(pwffn)
    res_pwffn = res_a + pwffn
    res_pwffn = t.pwn(res_pwffn)
    res_pwffn
end

function Base.show(io::IO, t::Transformer)
    hs = div(size(t.mh.iqproj.W)[1], t.mh.head)
    h, ps = size(t.pw.dout.W)

    print(io, "Transformer(")
    print(io, "head=$(t.mh.head), ")
    print(io, "head_size=$(hs), ")
    print(io, "pwffn_size=$(ps), ")
    print(io, "size=$(h)")
    if Flux.istraining()
        print(io, ", dropout=$(t.drop.p))")
    else
        print(io, ")")
    end
end

struct TransformerDecoder{MA<:MultiheadAttention, LA<:LayerNorm,
                          IMA<:MultiheadAttention, ILA<:LayerNorm,
                          P<:PwFFN, LP<:LayerNorm, DP<:Dropout} <: AbstractTransformer
    mh::MA
    mhn::LA
    imh::IMA
    imhn::ILA
    pw::P
    pwn::LP
    drop::DP
end

@functor TransformerDecoder

"""
    TransformerDecoder(size::Int, head::Int, ps::Int; act = relu, pdrop = 0.1)
    TransformerDecoder(size::Int, head::Int, hs::Int, ps::Int; act = relu, pdrop = 0.1)

TransformerDecoder layer. Decode the value from a Encoder.

`size` is the input size. if `hs` is not specify, use `div(size, head)` as the hidden size of multi-head attention. 
`ps` is the hidden size & `act` is the activation function of the positionwise feedforward layer. 
`pdrop` is the dropout rate.
"""
function TransformerDecoder(size::Int, head::Int, ps::Int; act = relu, pdrop = 0.1)
    rem(size, head) != 0 && error("size not divisible by head")
    TransformerDecoder(size, head, div(size, head), ps; act=act, pdrop=pdrop)
end

TransformerDecoder(size::Int, head::Int, hs::Int, ps::Int; act = relu, pdrop = 0.1) = TransformerDecoder(
    MultiheadAttention(head, size, hs, size; future=false, pdrop=pdrop),
    LayerNorm(size),
    MultiheadAttention(head, size, hs, size; future=true, pdrop=pdrop),
    LayerNorm(size),
    PwFFN(size, ps, act),
    LayerNorm(size),
    Dropout(pdrop),
)

function (td::TransformerDecoder)(x::AbstractArray{T,N}, m, mask=nothing) where {T,N}
    dropout = td.drop
    a = td.mh(x,x,x)
    a = dropout(a)
    res_a = x + a
    res_a = td.mhn(res_a)

    ia = td.imh(res_a, m, m, mask=mask)
    ia = dropout(ia)
    res_ia = res_a + ia
    res_ia = td.imhn(res_ia)

    pwffn = td.pw(res_ia)
    pwffn = dropout(pwffn)
    res_pwffn = res_ia + pwffn
    res_pwffn = td.pwn(res_pwffn)
    res_pwffn
end

function Base.show(io::IO, td::TransformerDecoder)
    hs = div(size(td.imh.iqproj.W)[1], td.imh.head)
    h, ps = size(td.pw.dout.W)

    print(io, "TransformerDecoder(")
    print(io, "head=$(td.mh.head), ")
    print(io, "head_size=$(hs), ")
    print(io, "pwffn_size=$(ps), ")
    print(io, "size=$(h)")
    if Flux.istraining()
        print(io, ", dropout=$(td.drop.p))")
    else
        print(io, ")")
    end
end
