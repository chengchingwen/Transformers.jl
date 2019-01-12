using Flux: @treelike, onehotbatch
import Flux: gpu

const Container = Union{NTuple{N, Vector{T}}, Vector{Vector{T}}} where N where T

function getmask(ls)
    lens = map(length, ls)
    m = zeros(maximum(lens), length(lens))

    for (i, l) ∈ enumerate(ls)
        selectdim(selectdim(m, 2, i), 1, 1:length(l)) .= 1
    end
    reshape(m, (1, size(m)...))
end

getmask(m1, m2) = permutedims_hack(m1, [2,1,3]) .* m2

struct Embed
    vocab
    unk
    embedding
end

@treelike Embed
gpu(e::Embed) = Embed(e.vocab, e.unk, gpu(e.embedding))

function Embed(size::Int, vocab, unk="</unk>")
    if !(unk ∈ vocab)
        push!(vocab, unk)
    end

    Embed(vocab, unk, param(randn(size, length(vocab))))
end

(e::Embed)(x) = e.embedding * device(onehotbatch(x, e.vocab, e.unk)), device(fill(1.0, (1, length(x))))
function (e::Embed)(xs::Container)
    maxlen = maximum(map(length, xs))
    cat([e.embedding * device(onehotbatch([x; fill(e.unk, maxlen - length(x))], e.vocab, e.unk)) for x ∈ xs]...;dims=3), device(getmask(xs))
end

onehot(e::Embed, x) = device(onehotbatch(x, e.vocab, e.unk))
function onehot(e::Embed, xs::Container)
    maxlen = maximum(map(length, xs))
    device(cat([onehotbatch([x; fill(e.unk, maxlen - length(x))], e.vocab, e.unk) for x ∈ xs]...;dims=3))
end

Base.show(io::IO, e::Embed) = print(io, "Embed($(size(e.embedding)[1]), vocab_size=$(length(e.vocab)), unk=$(e.unk))")
