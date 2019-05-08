"""
    Embed(size::Int, vocab_size::Int)

The Embedding Layer, `size` is the hidden size. `vocab_size` is the number of the vocabulary. Just a wrapper for embedding matrix.
"""
struct Embed{F ,W <: AbstractArray{F}}
    embedding::W
end

@treelike Embed

Base.size(e::Embed, s...) = size(e.embedding, s...)

Embed(size::Int, vocab_size::Int) = Embed(param(randn(Float32, size, vocab_size)))

(e::Embed)(x::AbstractArray{Int}) = gather(e.embedding, x)
(e::Embed{F})(x, scale) where {F} = e(x) .* convert(F, scale)

Base.show(io::IO, e::Embed) = print(io, "Embed($(size(e.embedding, 1)))")
