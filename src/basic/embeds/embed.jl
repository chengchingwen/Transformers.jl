"""
    Embed(size::Int, vocab_size::Int)

The Embedding Layer, `size` is the hidden size. `vocab_size` is the number of the vocabulary. Just a wrapper for embedding matrix.
"""
struct Embed{F ,W <: AbstractArray{F}} <: AbstractEmbed{F}
    scale::F
    embedding::W
end

Flux.functor(e::Embed) = (e.embedding,), m -> Embed(e.scale, m...)

Base.size(e::Embed, s...) = size(e.embedding, s...)

Embed(size::Int, vocab_size::Int; scale = one(Float32)) = Embed(Float32(scale), randn(Float32, size, vocab_size))

function (e::Embed)(x)
    if isone(e.scale)
        gather(e.embedding, x)
    else
        e(x, e.scale)
    end
end

(e::Embed{F})(x, scale) where {F} = gather(e.embedding, x) .* convert(F, scale)

Base.show(io::IO, e::Embed) = print(io, "Embed($(size(e.embedding, 1)))")
