"""
    Embed(size::Int, vocab_size::Int)

The Embedding Layer, `size` is the hidden size. `vocab_size` is the number of the vocabulary. Just a wrapper for embedding matrix.
"""
struct Embed{F ,W <: AbstractArray{F}} <: AbstractEmbed{F}
    scale::F
    embedding::W
end

@functor Embed

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

Base.show(io::IO, e::Embed) = print(io, "Embed(scale=$(e.scale), $(size(e.embedding, 1)))")

struct EmbeddingDecoder{E<:AbstractEmbed}
    embedding_layer::E
end
@functor EmbeddingDecoder

function embed_decode(embedding, x)
    x′ = reshape(x, size(x, 1), :)
    y = embedding' * x′
    return reshape(y, :, Base.tail(size(x))...)
end

(d::EmbeddingDecoder)(x) = embed_decode(d.embedding_layer.embedding, x)
function (d::EmbeddingDecoder{<:Embed})(x)
    y = embed_decode(d.embedding_layer.embedding, x)
    if isone(d.embedding_layer.scale)
        return y
    else
        return y .* convert(eltype(d.embedding_layer), d.embedding_layer.scale)
    end
end

Base.show(io::IO, d::EmbeddingDecoder) = (print(io, "EmbeddingDecoder("); show(io, d.embedding_layer); print(io, ')'))
