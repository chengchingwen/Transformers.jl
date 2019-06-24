struct WordPiece
  vocab::Vector{String}
  unk_idx::Int
  max_char::Int
  WordPiece(vocab::Vector{String}, unk_idx::Int , max_char::Int) = new(vocab, unk_idx, max_char)
end

WordPiece(vocab::Vector{String}, unk::String = "[UNK]"; max_char::Int=200) = WordPiece(vocab, findfirst(isequal(unk), vocab), max_char)

struct _wp_equal{first} <: Function
  ss::String
  base::Int
  bound::Int
  _wp_equal{T}(ss, base, bound) where T = new{T}(ss, base - 1, bound)
end

function (wq::_wp_equal{first})(s) where first
  if first
    start = 1
  else
    start = 3
  end

  fin = wq.bound - wq.base #length(wq.ss)
  len = length(s)

  if first
    fin != len && return false
  else
    @inbounds if !(s[1] == '#' == s[2])
      return false
    end
    fin != len - 2 && return false
  end

  @inbounds for i = start:len
    if first
      wq.ss[wq.base + i] != s[i] && return false
    else
      wq.ss[wq.base + i - 2] != s[i] && return false
    end
  end
  return true
end

(wp::WordPiece)(token) = wp(Vector{String}(), token)

function (wp::WordPiece)(tokens::Vector{T}) where T
  tks = Vector{T}()
  for tk ∈ tokens
    wp(tks, tk)
  end
  tks
end

function (wp::WordPiece)(tks::Vector{T}, token) where T
  s = 1
  tok_len = length(token)
  subtok = Vector{Int}()

  if tok_len <= wp.max_char
    failed = false
    while s <= tok_len
      e = tok_len
      failed = true
      while s < e
        if s != 1
          ss = findfirst(_wp_equal{false}(token, s, e), wp.vocab)
        else
          ss = findfirst(_wp_equal{true}(token, s, e), wp.vocab)
        end

        if ss === nothing
          e -= 1
        else
          push!(subtok, ss)
          failed = false
          s = e + 1
          break
        end
      end

      failed && break
    end
  else
    failed = true
  end

  if !failed
    len = length(tks)
    resize!(tks, len+length(subtok))
    for (i, tokid) ∈ enumerate(subtok)
      if T === Int
        @inbounds tks[len + i] = tokid
      else
        @inbounds tks[len + i] = wp.vocab[tokid]
      end
    end
  else
    if T === Int
      push!(tks, wp.unk_idx)
    else
      @inbounds push!(tks, wp.vocab[wp.unk_idx])
    end
  end

  tks
end
