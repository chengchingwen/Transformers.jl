struct WordPiece
  vocab::Vector{String}
  unk_idx::Int
  max_char::Int
  WordPiece(vocab::Vector{String}, unk_idx::Int , max_char::Int) = new(vocab, unk_idx, max_char)
end

Base.show(io::IO, wp::WordPiece) = print(io, "WordPiece(vocab_size=$(length(wp.vocab)), unk=$(wp.vocab[wp.unk_idx]), max_char=$(wp.max_char))")


"""
    WordPiece(vocab::Vector{String}, unk::String = "[UNK]"; max_char::Int=200)

WordPiece implementation.

    (wp::WordPiece)(token)

split given token.

    (wp::WordPiece)(tokens::Vector{String})

split given tokens

    (wp::WordPiece)(type, tokens::Vector{String})

split given tokens, if `type` is `Int`, return pieces indices instead of strings pieces.

    (wp::WordPiece)(tks::Vector{T}, token::String)

split given token and add result to `tks`. if `T` is `Int`, add indices instead of strings pieces.
"""
WordPiece(vocab::Vector{String}, unk::String = "[UNK]"; max_char::Int=200) = WordPiece(vocab, findfirst(isequal(unk), vocab), max_char)

Basic.Vocabulary(wp::WordPiece) = Vocabulary(wp.vocab, wp.vocab[wp.unk_idx])

struct _wp_equal{first} <: Function
  ss::String
  base::Int
  bound::Int
  ncode::Int
  _wp_equal{T}(ss, base, bound) where T = new{T}(ss, base, bound, getstrind(ss, bound+1)-getstrind(ss, base))
end

function getstrind(s, n)
  i = 1
  @inbounds while n > 1
    i = nextind(s, i)
    n-=1
  end
  i
end

_cmp(x, y, xbase, ybase, len) = ccall(:memcmp, Int32, (Ptr{UInt8}, Ptr{UInt8}, UInt),
                                      pointer(x, xbase),
                                      pointer(y, ybase),
                                      len)

function (wq::_wp_equal{first})(s) where first
  if first
    start = 1
  else
    start = 3
    iszero(_cmp(s, "##", 1, 1, 2)) || return false
  end

  wq.bound - wq.base == length(s) - start || return false
  return iszero(_cmp(wq.ss, s,
                   getstrind(wq.ss, wq.base),
                   start,
                   wq.ncode
                   ))
end


(wp::WordPiece)(token) = wp(Vector{String}(), token)

(wp::WordPiece)(tokens::Vector{String}) = wp(String, tokens)

function (wp::WordPiece)(type::Type{T}, tokens::Vector{String}) where T
  tks = Vector{T}()
  sizehint!(tks, length(tokens))
  for tk ∈ tokens
    wp(tks, tk)
  end
  tks
end

function (wp::WordPiece)(tks::Vector{T}, token::String) where T
  s = 1
  tok_len = length(token)
  subtok = Vector{Int}()

  sizehint!(subtok, 1)

  if tok_len <= wp.max_char
    failed = false
    while s <= tok_len
      e = tok_len
      failed = true
      while s <= e
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
