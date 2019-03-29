import Base

struct Vocabulary{T}
    siz::Int
    list::Vector{T}
    unk::T
    unki::Int
    function Vocabulary{T}(voc::Vector{T}, unk::T) where T
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

encode(vocab::Vocabulary{T}, i::T) where T = something(findfirst(isequal(l), vocab.list), vocab.unki)


encode(vocab::Vocabulary{T}, xs::Container{T}) where T = indices = map(x->encode(vocab, x), xs)

function encode(vocab::Vocabulary{T}, xs::Container{Container{T}}) where T
    indices = fill(vocab.unki, length(xs[1]), length(xs))
    for (i, x) ∈ enumerate(xs)
        for (j, xi) ∈ enumerate(x)
            @inbounds indices[j, i] = encode(vocab, xi)
        end
    end
    indices
end

(vocab::Vocabulary)(x) = encode(vocab, x)
