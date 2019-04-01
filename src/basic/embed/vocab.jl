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

Base.length(v::Vocabulary) = v.siz
Base.getindex(v::Vocabulary, i::Integer) = v.list[i]

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

(vocab::Vocabulary)(x) = encode(vocab, x)
(vocab::Vocabulary)(x, xs...) = (encode(vocab, x), map(x->encode(vocab, x), xs)...)

decode(vocab::Vocabulary{T}, i::Int) where T = 0 <= i <= length(vocab) ? vocab[i] : vocab.unk

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
end

Base.show(io::IO, v::Vocabulary) = print(io, "Vocabulary($(v.siz), unk=$(v.unk))")
