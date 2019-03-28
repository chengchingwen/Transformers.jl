using Flux: @treelike, onehotbatch
import Flux: gpu

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
    vocab::Vector{T}
    unk::T
    embedding::W
end

@treelike Embed
gpu(e::Embed) = Embed(e.vocab, e.unk, gpu(e.embedding))

function Embed(size::Int, vocab, unk="</unk>")
    if !(unk ∈ vocab)
        push!(vocab, unk)
    end

    Embed(vocab, unk, param(randn(Float32, size, length(vocab))))
end

(e::Embed)(x::Container) = e.embedding * device(onehotbatch(x, e.vocab, e.unk)), device(fill(1.0, (1, length(x))))::AbstractMatrix
function (e::Embed)(xs::Container{Vector{T}}) where T
    maxlen = maximum(map(length, xs))
    cat([e.embedding * device(onehotbatch([x; fill(e.unk, maxlen - length(x))], e.vocab, e.unk)) for x ∈ xs]...;dims=3)::ThreeDimArray, device(getmask(xs))::ThreeDimArray
end

onehot(e::Embed, x::Container) = device(onehotbatch(x, e.vocab, e.unk))::AbstractMatrix
function onehot(e::Embed, xs::Container{Vector{T}}) where T
    maxlen = maximum(map(length, xs))
    device(cat([onehotbatch([x; fill(e.unk, maxlen - length(x))], e.vocab, e.unk) for x ∈ xs]...;dims=3))::ThreeDimArray
end

Base.show(io::IO, e::Embed) = print(io, "Embed($(size(e.embedding)[1]), vocab_size=$(length(e.vocab)), unk=$(e.unk))")
