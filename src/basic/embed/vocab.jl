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


Base.show(io::IO, v::Vocabulary) = print(io, "Vocabulary($(v.siz), unk=$(v.unk))")
