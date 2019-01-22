using Flux
using Flux: @treelike

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
    drop::Dropout
end

@treelike Transformer

function Transformer(size::Int, head::Int, ps::Int; future::Bool = true, act = relu, pdrop = 0.1)
    rem(size, head) != 0 && error("size not divisible by head")
    Transformer(size, head, div(size, head), ps;future=future, act=act, pdrop=pdrop)
end

Transformer(size::Int, head::Int, hs::Int, ps::Int; future::Bool = true, act = relu, pdrop = 0.1) = Transformer(
    MultiheadAttention(head, size, hs, size; future=future),
    LayerNorm(size),
    Positionwise(size, ps, act),
    LayerNorm(size),
    Dropout(pdrop),
)

function (t::Transformer)(x, mask=nothing)
    a = t.mh(x, x, x; mask=mask)
    a = t.drop(a)
    n1 = t.LN1(x+a) # residual
    p = t.pw(n1)
    p = t.drop(p)
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
    print(io, "size=$(h)")
    if t.drop.active
        print(io, ", dropout=$(t.drop.p))")
    else
        print(io, ")")
    end
end

struct TransformerDecoder
    mhm::MultiheadAttention
    LN1::LayerNorm
    mh::MultiheadAttention
    LN2::LayerNorm
    pw::Positionwise
    LN3::LayerNorm
    drop::Dropout
end

@treelike TransformerDecoder

TransformerDecoder(size, head, hs, ps; act = relu, pdrop = 0.1) = TransformerDecoder(
    MultiheadAttention(head, size, hs, size; future=false),
    LayerNorm(size),
    MultiheadAttention(head, size, hs, size; future=true),
    LayerNorm(size),
    Positionwise(size, ps, act),
    LayerNorm(size),
    Dropout(pdrop),
)

function (td::TransformerDecoder)(x, m, mask=nothing)
    a1 = td.mhm(x,x,x)
    a1 = td.drop(a1)
    n1 = td.LN1(x+a1) # residual
    a = td.mh(n1, m, m, mask=mask)
    a = td.drop(a)
    n2 = td.LN2(a+n1) # residual
    p = td.pw(n2)
    p = td.drop(p)
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
    print(io, "size=$(h)")
    if t.drop.active
        print(io, ", dropout=$(t.drop.p))")
    else
        print(io, ")")
    end
end
