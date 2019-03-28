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

function (pw::Positionwise)(x)::AbstractMatrix
    # size(x) == (dims, seq_len)
    pw.dout(pw.din(x))
end

struct Transformer
    mh::MultiheadAttention
    mhn::LayerNorm
    pw::Positionwise
    pwn::LayerNorm
    drop::Dropout
end

@treelike Transformer

function Transformer(size::Int, head::Int, ps::Int; future::Bool = true, act = relu, pdrop = 0.1)
    rem(size, head) != 0 && error("size not divisible by head")
    Transformer(size, head, div(size, head), ps;future=future, act=act, pdrop=pdrop)
end

Transformer(size::Int, head::Int, hs::Int, ps::Int; future::Bool = true, act = relu, pdrop = 0.1) = Transformer(
    MultiheadAttention(head, size, hs, size; future=future, pdrop=pdrop),
    LayerNorm(size),
    Positionwise(size, ps, act),
    LayerNorm(size),
    Dropout(pdrop),
)

function (t::Transformer)(x::AbstractArray{T, N}, mask=nothing) where {T, N}
    a = t.mh(x, x, x; mask=mask)
    a = t.drop(a)
    res_a = x .+ a
    if N == 3
        insize = size(res_a)
        res_a = reshape(res_a, insize[1], :)
    end
    res_a = t.mhn(res_a)
    pwffn = t.pw(res_a)
    pwffn = t.drop(pwffn)
    res_pwffn = res_a .+ pwffn
    res_pwffn = t.pwn(res_pwffn)
    if N == 3
        res_pwffn = reshape(res_pwffn, :, Base.tail(insize)...)
    end
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
    if t.drop.active
        print(io, ", dropout=$(t.drop.p))")
    else
        print(io, ")")
    end
end

struct TransformerDecoder
    mh::MultiheadAttention
    mhn::LayerNorm
    imh::MultiheadAttention
    imhn::LayerNorm
    pw::Positionwise
    pwn::LayerNorm
    drop::Dropout
end

@treelike TransformerDecoder

TransformerDecoder(size, head, hs, ps; act = relu, pdrop = 0.1) = TransformerDecoder(
    MultiheadAttention(head, size, hs, size; future=false, pdrop=pdrop),
    LayerNorm(size),
    MultiheadAttention(head, size, hs, size; future=true, pdrop=pdrop),
    LayerNorm(size),
    Positionwise(size, ps, act),
    LayerNorm(size),
    Dropout(pdrop),
)

function (td::TransformerDecoder)(x::AbstractArray{T,N}, m, mask=nothing) where {T,N}
    a = td.mh(x,x,x)
    a = td.drop(a)
    res_a = x .+ a
    res_a = N == 3 ? @toNd(td.mhn(res_a)) : td.mhn(res_a)

    ia = td.imh(res_a, m, m, mask=mask)
    ia = td.drop(ia)
    res_ia = res_a .+ ia
    if N == 3
        insize = size(res_ia)
        res_ia = reshape(res_ia, insize[1], :)
    end
    res_ia = td.imhn(res_ia)
    pwffn = td.pw(res_ia)
    pwffn = td.drop(pwffn)
    res_pwffn = res_ia .+ pwffn
    res_pwffn = td.pwn(res_pwffn)
    if N == 3
        res_pwffn = reshape(res_pwffn, :, Base.tail(insize)...)
    end

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
    if td.drop.active
        print(io, ", dropout=$(td.drop.p))")
    else
        print(io, ")")
    end
end
