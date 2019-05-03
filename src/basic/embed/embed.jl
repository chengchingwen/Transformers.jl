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

Base.show(io::IO, e::Embed) = print(io, "Embed($(size(e.embedding, 1)))")
