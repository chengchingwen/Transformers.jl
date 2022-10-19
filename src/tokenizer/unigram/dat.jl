using Base64

struct DoubleArrayTrie
    array::Vector{UInt}
end

function common_prefix_search(dat::DoubleArrayTrie, key::AbstractVector{UInt8})
    node_pos = 0
    results = Int[]
    unit = dat.array[node_pos+1]
    node_pos ⊻= (unit >> 10) << ((unit & (UInt(1) << 9)) >> 6)
    for c in key
        iszero(c) && break
        node_pos ⊻= c
        unit = dat.array[node_pos+1]
        unit & ((UInt(1) << 31) | 0xff) != c && return results
        node_pos ⊻= (unit >> 10) << ((unit & (UInt(1) << 9)) >> 6)
        (unit >> 8) & 1 == 1 && push!(results, Int(dat.array[node_pos+1] & ((UInt(1) << 31) - 1)))
    end
    return results
end

function first_prefix_search(dat::DoubleArrayTrie, key::AbstractVector{UInt8})
    node_pos = 0
    unit = @inbounds dat.array[node_pos+1]
    node_pos ⊻= (unit >> 10) << ((unit & (UInt(1) << 9)) >> 6)
    for c in key
        iszero(c) && break
        node_pos ⊻= c
        unit = @inbounds dat.array[node_pos+1]
        unit & ((UInt(1) << 31) | 0xff) != c && return nothing
        node_pos ⊻= (unit >> 10) << ((unit & (UInt(1) << 9)) >> 6)
        (unit >> 8) & 1 == 1 && return @inbounds Int(dat.array[node_pos+1] & ((UInt(1) << 31) - 1))
    end
    return nothing
end

read32(data::AbstractVector{UInt8}, index=1) =
    UInt32(data[index+3]) << 24 |
    UInt32(data[index+2]) << 16 |
    UInt32(data[index+1]) << 8  |
    UInt32(data[index])

parse_charsmap(precompiled_charsmap::String) = parse_charsmap(base64decode(precompiled_charsmap))
function parse_charsmap(precompiled_charsmap::AbstractVector{UInt8})
    trie_size = read32(precompiled_charsmap)
    trie_char_size = div(trie_size, 4)
    trie_blob = Vector{UInt32}(undef, trie_char_size)
    for i in 1:trie_char_size
        n = read32(precompiled_charsmap, 4i+1)
        trie_blob[i] = n
    end
    normalized_blob = precompiled_charsmap[4trie_char_size+5:end]
    return normalized_blob, trie_blob
end

struct Precompiled
    precompiled_charsmap::Vector{UInt8}
    normalized::String
    length::Int
    trie::DoubleArrayTrie
end

Precompiled(precompiled_charsmap::String) = Precompiled(base64decode(precompiled_charsmap))
function Precompiled(precompiled_charsmap)
    normalized_blob, trie_blob = parse_charsmap(precompiled_charsmap)
    normalized = String(normalized_blob)
    trie = DoubleArrayTrie(trie_blob)
    return Precompiled(precompiled_charsmap, normalized, length(normalized), trie)
end

Base.show(io::IO, ::Precompiled) = print(io, "Precompiled(...)")

function _transform(normalized_bytes, len, index2)
    @inbounds while index2 <= len
        iszero(normalized_bytes[index2]) && break
        index2 += 1
    end
    return index2
end

transform(precompiled::Precompiled, chunk::AbstractString) = transform(precompiled, codeunits(chunk))
function transform(precompiled::Precompiled, chunk)
    result = first_prefix_search(precompiled.trie, chunk)
    isnothing(result) && return nothing
    index = result + 1
    index2 = index
    normalized_bytes = codeunits(precompiled.normalized)
    len = precompiled.length
    index2 = _transform(normalized_bytes, len, index2)
    normalized = (index, index2-1)
    return normalized
end

function (precompiled::Precompiled)(x)
    return _normalize(precompiled, x)
end

_normalize(precompiled, x) = _normalize!(IOBuffer(; sizehint = ncodeunits(x)), precompiled, x)
function _normalize!(data, precompiled, x)
    normalized = codeunits(precompiled.normalized)
    @inbounds for grapheme in Base.Unicode.graphemes(x)
        graphcode = codeunits(grapheme)
        if length(graphcode) < 6
            norm = transform(precompiled, graphcode)
            has_transform = !isnothing(norm)
            if has_transform
                n1, n2 = norm
                write(data, @view normalized[n1:n2])
                continue
            end
        end
        #=
        buf_is_empty = true
        for c in grapheme
          if c can be normalized
            if !buf_is_empty
              write buf to data
              buf_is_empty = true
            end
            write c to data
          else
            put c to buf
            buf_is_empty = false
          end
        end
        write buf to data
        =#

        itr = eachindex(grapheme)
        last_idx = length(graphcode) + 1
        char_index, char_end = iterate(itr)
        buf_is_empty = true
        write_char_index = char_index
        write_char_end = 0
        part = @view graphcode[char_index:char_end-1]
        norm = transform(precompiled, part)
        can_transform = !isnothing(norm)
        if can_transform
            n1, n2 = norm
            write(data, @view normalized[n1:n2])
            write_char_index = char_end
        else
            write_char_end = char_end
            buf_is_empty = false
        end
        state = iterate(itr, char_end)
        while !isnothing(state)
            char_index, char_end = state
            part = @view graphcode[char_index:char_end-1]
            norm = transform(precompiled, part)
            can_transform = !isnothing(norm)
            if can_transform
                if !buf_is_empty
                    write(data, @view graphcode[write_char_index:write_char_end-1])
                    buf_is_empty = true
                    write_char_end = char_end
                end
                n1, n2 = norm
                write(data, @view normalized[n1:n2])
                write_char_index = char_end
            else
                write_char_end = char_end
                buf_is_empty = false
            end
            state = iterate(itr, char_end)
        end

        if !buf_is_empty
            write(data, @view graphcode[write_char_index:write_char_end-1])
        end
    end
    return String(take!(data))
end
