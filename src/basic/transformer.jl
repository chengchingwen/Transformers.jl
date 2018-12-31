using Flux
using Flux: @treelike

#extend Flux LayerNorm for 3-dims input
function (a::LayerNorm)(x::ThreeDimArray{T}) where T
    s = size(x)
    reshape(a(reshape(x, s[1], :)), s)
end

struct Positionwise
    din::Dense
    dout::Dense
end

@treelike Positionwise

Positionwise(size::Int, h::Int, act = relu) = Positionwise(
    Dense(size, h, act),
    Dense(h, size)
)

function (pw::Positionwise)(x)
    # size(x) == (dims, seq_len)
    pw.dout(pw.din(x))
end

function (pw::Positionwise)(x::ThreeDimArray{T}) where T
    s = size(x)
    reshape(pw.dout(pw.din(reshape(x, s[1], :))), s)
end

struct Transformer
    mh::MultiheadAttention
    LN1::LayerNorm
    pw::Positionwise
    LN2::LayerNorm
end

@treelike Transformer

function Transformer(size::Int, head::Int, ps::Int; future::Bool = true, act = relu)
    rem(size, head) != 0 && error("size not divisible by head")
    Transformer(size, head, div(size, head), ps;future=future, act=act)
end

Transformer(size::Int, head::Int, hs::Int, ps::Int; future::Bool = true, act = relu) = device(Transformer(
    MultiheadAttention(head, size, hs, size; future=future),
    LayerNorm(size),
    Positionwise(size, ps, act),
    LayerNorm(size)
))

function (t::Transformer)(x, mask=nothing)
    a = t.mh(x, x, x; mask=mask)
    n1 = t.LN1(x+a) # residual
    p = t.pw(n1)
    n2 = t.LN2(p+n1) # residual
    n2
end

function Base.show(io::IO, t::Transformer)
    hs = div(size(t.mh.iqproj.W)[1], t.mh.head)
    h, ps = size(t.pw.dout.W)

    print(io, "Transformer(")
    print(io, "head=$(t.mh.head), ")
    print(io, "head_size=$(hs), ")
    print(io, "pwffn_size=$(ps), ")
    print(io, "size=$(h))")
end

struct TransformerDecoder
    mhm::MultiheadAttention
    LN1::LayerNorm
    mh::MultiheadAttention
    LN2::LayerNorm
    pw::Positionwise
    LN3::LayerNorm
end

@treelike TransformerDecoder

TransformerDecoder(size, head, hs, ps; act = relu) = device(TransformerDecoder(
    MultiheadAttention(head, size, hs, size; future=false),
    LayerNorm(size),
    MultiheadAttention(head, size, hs, size; future=true),
    LayerNorm(size),
    Positionwise(size, ps, act),
    LayerNorm(size)
))

function (td::TransformerDecoder)(x, m, mask=nothing)
    a1 = td.mhm(x,x,x)
    n1 = td.LN1(x+a1) # residual
    a = td.mh(n1, m, m, mask=mask)
    n2 = td.LN2(a+n1) # residual
    p = td.pw(n2)
    n3 = td.LN3(p+n2) # residual
    n3
end

function Base.show(io::IO, t::TransformerDecoder)
    hs = div(size(t.mh.iqproj.W)[1], t.mh.head)
    h, ps = size(t.pw.dout.W)

    print(io, "TransformerDecoder(")
    print(io, "head=$(t.mhm.head), ")
    print(io, "head_size=$(hs), ")
    print(io, "pwffn_size=$(ps), ")
    print(io, "size=$(h))")
end
