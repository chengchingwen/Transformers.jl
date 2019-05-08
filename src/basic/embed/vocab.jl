"""
    Vocabulary{T}(voc::Vector{T}, unk::T) where T

struct for holding the vocabulary list to encode/decode input tokens.
"""
struct Vocabulary{T}
    siz::Int
    list::Vector{T}
    unk::T
    unki::Int
    function Vocabulary(voc::Vector{T}, unk::T) where T
        if !(unk ∈ voc)
            pushfirst!(voc, unk)
            unki = 1
        else
            unki = findfirst(isequal(unk), voc)
        end

        new{T}(length(voc), voc, unk, unki)
    end
end

Base.length(vocab::Vocabulary) = vocab.siz
Base.getindex(vocab::Vocabulary, is) = decode(vocab, is)
Base.getindex(vocab::Vocabulary, i, is...) = (decode(vocab, i), map(i->decode(vocab, i), is)...)

"""
    encode(vocab::Vocabulary, x)

encode the given data to the index encoding.
"""
encode(vocab::Vocabulary{T}, i::Union{T,W}) where {T,W} = something(findfirst(isequal(i), vocab.list), vocab.unki)

encode(vocab::Vocabulary{T}, xs::Container{<:Union{T,W}}) where {T,W} = indices = map(x->encode(vocab, x), xs)

function encode(vocab::Vocabulary{T}, xs::Container{<:Container{<:Union{T,W}}}) where {T,W}
    lens = map(length, xs)
    indices = fill(vocab.unki, maximum(lens), length(xs))
    for (i, x) ∈ enumerate(xs)
        for (j, xi) ∈ enumerate(x)
            @inbounds indices[j, i] = encode(vocab, xi)
        end
    end
    indices
end

"""
    (vocab::Vocabulary)(x)

encode the given data to the index encoding.
"""
(vocab::Vocabulary)(x) = encode(vocab, x)
(vocab::Vocabulary)(x, xs...) = (encode(vocab, x), map(x->encode(vocab, x), xs)...)

decode(vocab::Vocabulary{T}, i::Int) where T = 0 <= i <= length(vocab) ? vocab.list[i] : vocab.unk

decode(vocab::Vocabulary{T}, is::Container{Int}) where T = map(i->decode(vocab, i), is)

function decode(vocab::Vocabulary{T}, is::Container{<:Container{Int}}) where T
    tokens = Vector{Vector{T}}(undef, length(is))
    for (idx, i) ∈ enumerate(is)
        token = decode(vocab, i)
        tokens[idx] = token
    end
    tokens
end

function decode(vocab::Vocabulary{T}, is::AbstractMatrix{Int}) where T
    ilen, olen = size(is)
    tokens = Vector{Vector{T}}(undef, olen)
    for idx ∈ 1:olen
        token = Vector{T}(undef, ilen)
        for idy ∈ 1:ilen
            token[idy] = decode(vocab, is[idy, idx])
        end
        tokens[idx] = token
    end
    tokens
end

Base.show(io::IO, v::Vocabulary) = print(io, "Vocabulary($(v.siz), unk=$(v.unk))")

Flux.onehot(v::Vocabulary{T}, x::Container{<:Union{T, Container{T}}}) where T = onehot(v, v(x))
Flux.onehot(v::Vocabulary, x::AbstractArray{Int}) = onehotarray(length(v), x)

Flux.onecold(v::Vocabulary{T}, p) where T = decode(v, _onecold(p))
function _onecold(p)
    p = collect(p)
    y = Array{Int}(undef, Base.tail(size(p)))
    for i = 1:length(y)
        ind = Tuple(CartesianIndices(y)[i])
        y[i] = argmax(@view(p[:, ind...]))
    end
    y
end

"""
    getmask(ls::Container{<:Container})

get the mask for batched data.
"""
function getmask(ls::Container{<:Container})
    lens = map(length, ls)
    m = zeros(Float32, maximum(lens), length(lens))

    for (i, l) ∈ enumerate(ls)
        selectdim(selectdim(m, 2, i), 1, 1:length(l)) .= 1
    end
    reshape(m, (1, size(m)...))
end

"""
    getmask(m1::A, m2::A) where A <: Abstract3DTensor

get the mask for the covariance matrix.
"""
getmask(m1::A, m2::A) where A <: Abstract3DTensor = permutedims(m1, [2,1,3]) .* m2

