using Flux
using Flux: @treelike

struct Positionwise
    din::Dense
    dout::Dense
end

@treelike Positionwise

Positionwise(size::Int, h::Int) = Positionwise(
    Dense(size, h, relu),
    Dense(h, size)
)

function (pw::Positionwise)(x)
    # size(x) == (dims, seq_len)
    pw.dout(pw.din(x))
end

struct Transformer
    mh::MultiheadAttention
    LN1::LayerNorm
    pw::Positionwise
    LN2::LayerNorm
end

@treelike Transformer

Transformer(size::Int, head::Int, hs::Int, ps::Int; future::Bool = true) = device(Transformer(
    MultiheadAttention(head, size, hs, size; future=future),
    LayerNorm(size),
    Positionwise(size, ps),
    LayerNorm(size)
))

function (t::Transformer)(x, mask=nothing)
    a = t.mh(x, x, x; mask=mask)
    n1 = t.LN1(x+a) # residual
    p = t.pw(n1)
    n2 = t.LN2(p+n1) # residual
    n2
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

TransformerDecoder(size, head, hs, ps) = device(TransformerDecoder(
    MultiheadAttention(head, size, hs, size; future=false),
    LayerNorm(size),
    MultiheadAttention(head, size, hs, size; future=true),
    LayerNorm(size),
    Positionwise(size, ps),
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
