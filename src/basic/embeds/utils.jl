using TextEncodeBase: with_head_tail

#=
 with_firsthead_tail(x, head, tail)

Apply `head` at the first sentence, and add `tail` to all the sentences.
This is for handling input of bert that have more than one segment.

Only works only concrete element typed input.

cases <input>:
 1. a single sentence: Input `Vector{String}`, Apply `with_head_tail`.
 2. batch of sentence: Input `Vector{Vector{String}}`, Apply `with_head_tail`.
 3. batch of multiple sentence: Input `Vector{Vector{Vector{String}}}`. Outer-most length is batch size, and
     middle length is the number of segment. Apply head to the first segment, and add `tail` to all segments.
=#
with_firsthead_tail(x::AbstractVector, head, tail, sep = tail) =
    map(with_firsthead_tail(head, tail, sep), x)
with_firsthead_tail(x::AbstractVector{<:AbstractString}, head, tail, sep = tail) = with_head_tail(x, head, tail)
with_firsthead_tail(x::AbstractVector{<:AbstractVector{<:AbstractString}}, head, tail, sep = tail) =
    with_head_tail(x, head, tail)
function with_firsthead_tail(x::AbstractVector{<:AbstractVector{<:AbstractVector{S}}},
                             head, tail, sep = tail) where S<:AbstractString
    n = length(x)
    T = promote_type(S, typeof(head), typeof(tail))
    vecs = Vector{Vector{Vector{T}}}(undef, n); empty!(vecs)
    for doc in x
        len = length(doc)
        vec = Vector{Vector{T}}(undef, len)
        for (i, sent) in enumerate(doc)
            h = i == 1 ? head : nothing
            t = i == len ? tail : sep
            vec[i] = with_head_tail(sent, h, t)
        end
        push!(vecs, vec)
    end
    return vecs
end
with_firsthead_tail(head, tail, sep = tail) = TextEncodeBase.FixRest(with_firsthead_tail, head, tail, sep)
with_firsthead_tail(x; head=nothing, tail=nothing, sep = tail) = with_firsthead_tail(x, head, tail, sep)
with_firsthead_tail(; head=nothing, tail=nothing, sep = tail) = with_firsthead_tail(head, tail, sep)

#=
 segment_and_concat

Compute the segment for input of bert. If input have multiple segment, concate each segment to one array,
 and mark the segment number correspondingly. If input only have single segment, return the input and
 an array of 1.
=#
segment_and_concat(tok::AbstractVector) = map(segment_and_concat, tok)
segment_and_concat(tok::AbstractVector{<:AbstractString}) = tok, ones(Int, length(tok))
segment_and_concat(tok::AbstractVector{<:AbstractVector{<:AbstractString}}) = tok, map(t->segment_and_concat(t)[2], tok)
function segment_and_concat(tok::AbstractVector{<:AbstractVector{<:AbstractVector{T}}}) where T<:AbstractString
    N = length(tok)
    segments = Vector{Vector{Int}}(undef, N); empty!(segments)
    toks = Vector{Vector{T}}(undef, N); empty!(toks)
    for doc in tok
        n = sum(length, doc)
        segment = Vector{Int}(undef, n)
        words = Vector{T}(undef, n)
        offset = 1
        for (i, sent) in enumerate(doc)
            len = length(sent)
            copyto!(segment, offset, Iterators.repeated(i), 1, len)
            copyto!(words, offset, sent, 1, len)
            offset += len
        end
        push!(segments, segment)
        push!(toks, words)
    end
    return toks, segments
end

#=
 concat

concat the segment for input. If input have multiple segment, concate each segment to one array.
 If input only have single segment, return the input.
=#
concat(tok::AbstractVector) = map(concat, tok)
concat(tok::AbstractVector{<:AbstractString}) = tok
concat(tok::AbstractVector{<:AbstractVector{<:AbstractString}}) = tok
function concat(tok::AbstractVector{<:AbstractVector{<:AbstractVector{T}}}) where T<:AbstractString
    N = length(tok)
    toks = Vector{Vector{T}}(undef, N); empty!(toks)
    for doc in tok
        n = sum(length, doc)
        words = Vector{T}(undef, n)
        offset = 1
        for (i, sent) in enumerate(doc)
            len = length(sent)
            copyto!(words, offset, sent, 1, len)
            offset += len
        end
        push!(toks, words)
    end
    return toks
end
