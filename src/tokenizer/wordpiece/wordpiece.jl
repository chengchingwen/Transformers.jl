using StringViews

import DoubleArrayTries
using DoubleArrayTries: DoubleArrayTrie

const DAT = DoubleArrayTries

struct WordPiece
    trie::DoubleArrayTrie
    index::DAT.CVector
    unki::Int
    max_char::Int
    subword_prefix::String
end

function WordPiece(vocab::Vector{String}, unk::String = "[UNK]"; max_char = 200, subword_prefix = "##")
    trie = DoubleArrayTrie(copy(vocab))
    unki = DAT.lookup(trie, unk)
    index = Vector{Int}(undef, length(vocab))
    for (i, str) in enumerate(vocab)
        index[DAT.lookup(trie, str)] = i
    end
    return WordPiece(trie, DAT.CVector(index), unki, max_char, subword_prefix)
end
function WordPiece(vocab::Vector{String}, unki::Int; max_char = 200, subword_prefix = "##")
    @assert 0 < unki <= length(vocab)
    unk = vocab[unki]
    return WordPiece(vocab, unk; max_char, subword_prefix)
end

struct _WithPrefix{A1, A2} <: AbstractVector{UInt8}
    x::A1
    y::A2
    offset::Int
    length::Int
end
_WithPrefix(x, y) = (lenx = length(x); _WithPrefix(x, y, lenx, lenx + length(y)))
function Base.getindex(x::_WithPrefix, i)
    offset = x.offset
    return i > offset ? x.y[i - offset] : x.x[i]
end
Base.length(x::_WithPrefix) = x.length
Base.size(x::_WithPrefix) = (length(x),)

function (wp::WordPiece)(x)
    result = Vector{String}()
    isempty(x) && return result
    len = ncodeunits(x)
    failed = true
    s = 1
    if length(x) <= wp.max_char
        codes = codeunits(x)
        prefix = codeunits(wp.subword_prefix)
        while s <= len
            e = lastindex(x)
            failed = true
            while s <= e
                cbuf = @view codes[s:nextind(x, e) - 1]
                buf = isone(s) ? cbuf : _WithPrefix(prefix, cbuf)
                id = DAT.lookup(wp.trie, buf)
                if iszero(id)
                    e = prevind(x, e)
                else
                    push!(result, DAT.decode(wp.trie, id))
                    failed = false
                    s = nextind(x, e)
                    break
                end
            end
            failed && break
        end
    end

    if failed
        empty!(result)
        push!(result, DAT.decode(wp.trie, wp.unki))
    end
    return result
end
