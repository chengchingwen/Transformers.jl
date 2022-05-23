using TextEncodeBase
using TextEncodeBase: trunc_and_pad, nested2batch
import TextEncodeBase: encode, decode

"""
    Vocabulary{T}(voc::Vector{T}, unk::T) where T

struct for holding the vocabulary list to encode/decode input tokens.
"""
struct Vocabulary{T, V<:Vocab{T}} <: AbstractVocabulary{T}
    vocab::V
end

function Vocabulary(list::Vector{T}, unk::T) where T
    if unk ∉ list
        pushfirst!(list, unk)
    end
    return Vocabulary(Vocab(list, unk))
end

Base.length(vocab::Vocabulary) = length(vocab.vocab)
Base.getindex(vocab::Vocabulary, is) = decode(vocab, is)
Base.getindex(vocab::Vocabulary, i, is...) = (decode(vocab, i), map(i->decode(vocab, i), is)...)

check_vocab(vocab::Vocabulary, word) = check_vocab(vocab.vocab, word)
check_vocab(vocab::Vocab, word) = findfirst(==(word), vocab.list) !== nothing

"""
    encode(vocab::Vocabulary, x)

encode the given data to the index encoding.
"""
encode(vocab::Vocabulary, i) = lookup(Int, vocab.vocab, i)
encode(vocab::Vocabulary, i, is...) = (encode(vocab, i), map(Base.Fix1(encode, vocab), is)...)

function encode(vocab::Vocabulary{T}, xs::Container{<:Container{<:Union{T,W}}}) where {T,W}
    nested2batch(trunc_and_pad(lookup(Int, vocab.vocab, xs), nothing, vocab.vocab.unki))
end

"""
    (vocab::Vocabulary)(x)

encode the given data to the index encoding.
"""
(vocab::Vocabulary)(xs...) = encode(vocab, xs...)

decode(vocab::Vocabulary, i) = lookup(String, vocab.vocab, i)

function decode(vocab::Vocabulary, is::AbstractMatrix{Int})
    olen = size(is, 2)
    tokens = Vector{Vector{eltype(vocab)}}(undef, olen)
    for idx ∈ 1:olen
        tokens[idx] = lookup(String, vocab.vocab, @view is[begin:end, idx])
    end
    tokens
end

Base.eltype(v::Vocabulary{T}) where T = T
Base.show(io::IO, v::Vocabulary) = print(io, "Vocabulary{$(eltype(v))}($(length(v)), unk=$(v.vocab.unk))")

function Flux.onehot(v::Vocabulary, x)
  vt = eltype(v)
  xt = if typeof(x) <: Container || typeof(x) <: AbstractArray{Int}
    eltype(x)
  else
    typeof(x)
  end
  vsize = length(v)
  if xt === Int && vt !== Int
    return OneHotArray(vsize, x)
  else
    return OneHotArray(vsize, v(x))
  end
end

Flux.onecold(v::Vocabulary{T}, p) where T = decode(v, _onecold(p))

_onecold(p) = (map(argmax(p; dims=1)) do pi
  first(Tuple(pi))
end) |> Base.Fix2(reshape, Base.tail(size(p))) |> collect

"""
    getmask(ls::Container{<:Container})

get the mask for batched data.
"""
function getmask(ls::Container{<:Container}, n::Integer)
    m = zeros(Float32, n, length(ls))

    for (i, l) ∈ enumerate(ls)
        selectdim(selectdim(m, 2, i), 1, 1:min(length(l), n)) .= 1
    end
    reshape(m, (1, size(m)...))
end

getmask(ls::Container{<:Container}) = getmask(ls, maximum(length, ls))
getmask(v::Vector) = nothing
getmask(v::Vector, n::Integer) = nothing

"""
    getmask(m1::A, m2::A) where A <: Abstract3DTensor

get the mask for the covariance matrix.
"""
getmask(m1::A, m2::A) where A <: Abstract3DTensor = permutedims(m1, [2,1,3]) .* m2

