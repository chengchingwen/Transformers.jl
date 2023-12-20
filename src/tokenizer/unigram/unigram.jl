using DataStructures: MutableLinkedList
using StringViews
using LRUCache

import DoubleArrayTries
using DoubleArrayTries: DoubleArrayTrie, CommonPrefixSearch

const DAT = DoubleArrayTries

abstract type AbstractUnigram end

struct Unigram <: AbstractUnigram
    trie::DoubleArrayTrie
    unki::Int
    scores::Vector{Float64}
    min_score::Float64
end

function Unigram(vocab::Vector{String}, scores::Vector{Float64}, unki)
    @assert 0 < unki <= length(vocab)
    unk = vocab[unki]
    trie = DoubleArrayTrie(copy(vocab))
    unki = DAT.lookup(trie, unk)
    nscores = similar(scores)
    for (str, score) in zip(vocab, scores)
        id = DAT.lookup(trie, str)
        @assert id != 0
        nscores[id] = score
    end
    return Unigram(trie, unki, nscores, minimum(nscores))
end

(unigram::Unigram)(x) = optimized_encode(unigram, x)

Base.show(io::IO, unigram::Unigram) =
    print(io, "Unigram(vocab_size = ", length(unigram.trie), ", unk = ", DAT.decode(unigram.trie, unigram.unki), ')')

const kUnkPenalty = 10.0

BestNodePath(id = -1, best_path_score = 0.0, starts_at = -1) =
    (id = id, best_path_score = best_path_score, starts_at = starts_at)

function optimized_encode(unigram::Unigram, x)
    results = MutableLinkedList{String}()
    isempty(x) && return Vector{String}()
    size = ncodeunits(x)
    codes = codeunits(x)
    unk_score = unigram.min_score - kUnkPenalty
    best_path_ends_at = Vector{@NamedTuple{id::Int, best_path_score::Float64, starts_at::Int}}(undef, size + 1)
    fill!(best_path_ends_at, BestNodePath())
    starts_at = 1
    @inbounds while starts_at <= size
        best_path_score_till_here = best_path_ends_at[starts_at].best_path_score
        has_single_node = false
        mblen = ncodeunits(x[starts_at])
        for (id, token) in CommonPrefixSearch(unigram.trie, @view(codes[starts_at:end]))
            len = ncodeunits(token)
            key_pos = starts_at + len
            target_node = best_path_ends_at[key_pos]
            score = unigram.scores[id]
            candidate_best_path_score = score + best_path_score_till_here
            if target_node.starts_at == -1 || candidate_best_path_score > target_node.best_path_score
                target_node = BestNodePath(id, candidate_best_path_score, starts_at)
                best_path_ends_at[key_pos] = target_node
            end
            if !has_single_node && len == mblen
                has_single_node = true
            end
        end
        if !has_single_node
            key_pos = starts_at + mblen
            target_node = best_path_ends_at[key_pos]
            candidate_best_path_score = unk_score + best_path_score_till_here
            if target_node.starts_at == -1 || candidate_best_path_score > target_node.best_path_score
                target_node = BestNodePath(unigram.unki, candidate_best_path_score, starts_at)
                best_path_ends_at[key_pos] = target_node
            end
        end
        starts_at += mblen
    end
    ends_at = size+1
    unk_buf = false
    unk_end = size
    unk_start = -1
    @inbounds while ends_at > 1
        node = best_path_ends_at[ends_at]
        starts_at = node.starts_at
        if node.id == unigram.unki
            unk_start = starts_at
            unk_buf = true
        else
            if unk_buf
                pushfirst!(results, StringView(codes[unk_start:unk_end]))
                unk_buf = false
            end
            pushfirst!(results, DAT.decode(unigram.trie, node.id))
            unk_end = starts_at-1
        end
        ends_at = starts_at
    end
    return collect(results)
end

struct CachedUnigram{U <: AbstractUnigram, D <: AbstractDict{<:AbstractString, Vector{String}}} <: AbstractUnigram
    unigram::U
    cache::D
end
CachedUnigram(unigram::AbstractUnigram) = CachedUnigram(unigram, LRU{AbstractString, Vector{String}}(; maxsize = 1000))

(unigram::CachedUnigram)(x) = get!(()->unigram.unigram(x), unigram.cache, x)

Base.show(io::IO, unigram::CachedUnigram) = (print(io, "CachedUnigram("); show(io, unigram.unigram); print(io, ')'))
