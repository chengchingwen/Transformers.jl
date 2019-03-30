function getmask(ls)
    lens = map(length, ls)
    m = zeros(Float32, maximum(lens), length(lens))

    for (i, l) ∈ enumerate(ls)
        selectdim(selectdim(m, 2, i), 1, 1:length(l)) .= 1
    end
    reshape(m, (1, size(m)...))
end

getmask(m1, m2) = permutedims(m1, [2,1,3]) .* m2

struct Embed{W, T}
    Vocab::Vocabulary{T}
    embedding::W
end

@treelike Embed

Embed(size::Int, vocab::Vocabulary) = Embed(vocab, param(randn(Float32, size, length(vocab))))

(e::Embed)(x::AbstractArray{Int}) = e.embedding * onehotarray(length(e.Vocab), x)


(e::Embed)(x) = e.embedding * onehot(e, x)

Flux.onehot(e::Embed, x) = onehot(e.Vocab, x)
Flux.onehot(v::Vocabulary, x) = onehotarray(length(v), x)

Flux.onecold(e::Embed, p) = onecold(e.Vocab, p)
Flux.onecold(v::Vocabulary{T}, p) where T = map(i->v.list[i], p)
function Flux.onecold(p)
    y = similar{Int}(p, Base.tail(size(p)))
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
