using StringViews

struct Unigram
    vocab::Vector{String}
    unki::Int
    scores::Vector{Float64}
    min_score::Float64
    trie::Trie{UInt8, Int}
end

function Unigram(vocab::Vector{String}, scores::Vector{Float64}, unki)
    @assert 0 < unki <= length(vocab)
    trie = Trie(map(codeunits, vocab), 1:length(vocab))
    return Unigram(vocab, unki, scores, minimum(scores), trie)
end

(unigram::Unigram)(x) = optimized_encode(unigram, x)

Base.show(io::IO, unigram::Unigram) =
    print(io, "Unigram(vocab_size = ", length(unigram.vocab), ", unk = ", unigram.vocab[unigram.unki], ')')

const kUnkPenalty = 10.0

BestNodePath(id = -1, best_path_score = 0.0, starts_at = -1) =
    (id = id, best_path_score = best_path_score, starts_at = starts_at)

function optimized_encode(unigram::Unigram, x)
    results = Vector{String}()
    isempty(x) && return results
    size = ncodeunits(x)
    codes = codeunits(x)
    unk_score = unigram.min_score - kUnkPenalty
    best_path_ends_at = Vector{@NamedTuple{id::Int, best_path_score::Float64, starts_at::Int}}(undef, size + 1)
    fill!(best_path_ends_at, BestNodePath())
    starts_at = 1
    while starts_at <= size
        best_path_score_till_here = best_path_ends_at[starts_at].best_path_score
        has_single_node = false
        mblen = ncodeunits(x[starts_at])
        for token_bytes in find_prefixes(unigram.trie, @view(codes[starts_at:end]))
            len = length(token_bytes)
            token = StringView(token_bytes)
            key_pos = starts_at + len
            target_node = best_path_ends_at[key_pos]
            id = findfirst(==(token), unigram.vocab)
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
    while ends_at > 1
        node = best_path_ends_at[ends_at]
        starts_at = node.starts_at
        if node.id == unigram.unki
            unk_start = starts_at
            unk_buf = true
        else
            if unk_buf
                push!(results, StringView(codes[unk_start:unk_end]))
                unk_buf = false
            end
            push!(results, unigram.vocab[node.id])
            unk_end = starts_at-1
        end
        ends_at = starts_at
    end
    reverse!(results)
    return results
end
