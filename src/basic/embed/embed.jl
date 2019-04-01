function getmask(ls::Container{<:Container})
    lens = map(length, ls)
    m = zeros(Float32, maximum(lens), length(lens))

    for (i, l) ∈ enumerate(ls)
        selectdim(selectdim(m, 2, i), 1, 1:length(l)) .= 1
    end
    reshape(m, (1, size(m)...))
end

getmask(m1::A, m2::A) where A <: Abstract3DTensor = permutedims(m1, [2,1,3]) .* m2

struct Embed{F ,W <: AbstractArray{F}, T}
    Vocab::Vocabulary{T}
    embedding::W
end

@treelike Embed

Embed(size::Int, vocab::Vocabulary) = Embed(vocab, param(randn(Float32, size, length(vocab))))

(e::Embed)(x::AbstractArray{Int}) = e.embedding * onehotarray(length(e.Vocab), x)
(e::Embed{F,W,T})(x::Container{<:Union{T, Container{T}}}) where {F,W,T} = e.embedding * onehot(e, x)
(e::Embed{F})(x, scale) where {F} = e(x) .* convert(F, scale)

Flux.onehot(e::Embed{F, W, T}, x::Container{<:Union{T, Container{T}}}) where {F,W,T} = onehot(e, x)
Flux.onehot(v::Vocabulary{T}, x::Container{<:Union{T, Container{T}}}) where T = v(x)
Flux.onehot(e::Embed, x::AbstractArray{Int}) = onehot(e.Vocab, x)
Flux.onehot(v::Vocabulary, x::AbstractArray{Int}) = onehotarray(length(v), x)

Flux.onecold(e::Embed, p) = onecold(e.Vocab, p)
Flux.onecold(v::Vocabulary{T}, p) where T = decode(v, _onecold(p))

function _onecold(p)
    p = collect(p)
    y = Array{Int}(p, Base.tail(size(p)))
    for i = 1:length(y)
        ind = Tuple(CartesianIndices(y)[i])
        y[i] = argmax(@view(p[:, ind...]))
    end
    y
end

function Base.show(io::IO, e::Embed)
    print(io, "Embed($(size(e.embedding, 1)), ")
    show(io,  e.Vocab)
    print(io, ")")
end
