using InternedStrings
using TextEncodeBase: trunc_and_pad, nested2batch, with_head_tail, nestedcall, getvalue
using TextEncodeBase: WordTokenization

string_getvalue(x::TextEncodeBase.TokenStage) = intern(getvalue(x))::String

# text encoder

"""
    struct TransformerTextEncoder{T<:AbstractTokenizer, V<:AbstractVocabulary{String}, P} <: AbstractTextEncoder
        tokenizer::T
        vocab::V
        process::P
        startsym::String
        endsym::String
        padsym::String
        trunc::Union{Nothing, Int}
    end

The text encoder for general transformers. Taking a tokenizer, vocabulary, and a processing function, configured with
 a start symbol, an end symbol, a padding symbol, and a maximum length.

    TransformerTextEncoder(tokenze, vocab, process; trunc = nothing,
                           startsym = "<s>", endsym = "</s>", unksym = "<unk>", padsym = "<pad>")

`tokenize` can be any tokenize function from `WordTokenizers`. `vocab` is either a list of word or a `Vocab`.
 `process` can be omitted, then a predefined processing pipeline will be used.

    TransformerTextEncoder(f, e::TransformerTextEncoder)

Take a text encoder and create a new text encoder with same configuration except the processing function.
 `f` is a function that take the encoder and return a new process function. This is useful for changing part of
 the procssing function.

# Example
```julia-repl
julia> textenc = TransformerTextEncoder(labels; startsym, endsym, unksym,
                                        padsym = unksym, trunc = 100)
TransformerTextEncoder(
├─ TextTokenizer(default),
├─ vocab = Vocab{String, SizedArray}(size = 37678, unk = </unk>, unki = 1),
├─ startsym = <s>,
├─ endsym = </s>,
├─ padsym = </unk>,
├─ trunc = 100,
└─ process = Pipelines:
  ╰─ target[tok] := nestedcall(string_getvalue, source)
  ╰─ target[tok] := with_head_tail(<s>, </s>)(target.tok)
  ╰─ target[trunc_tok] := trunc_and_pad(100, </unk>)(target.tok)
  ╰─ target[trunc_len] := nestedmaxlength(target.trunc_tok)
  ╰─ target[mask] := getmask(target.tok, target.trunc_len)
  ╰─ target[tok] := nested2batch(target.trunc_tok)
  ╰─ target := (target.tok, target.mask)
)

julia> Basic.TransformerTextEncoder(textenc) do enc
           Pipelines(enc.process[1:4]) |> PipeGet{(:trunc_tok, :trunc_len)}()
       end
TransformerTextEncoder(
├─ TextTokenizer(default),
├─ vocab = Vocab{String, SizedArray}(size = 37678, unk = </unk>, unki = 1),
├─ startsym = <s>,
├─ endsym = </s>,
├─ padsym = </unk>,
├─ trunc = 100,
└─ process = Pipelines:
  ╰─ target[tok] := nestedcall(string_getvalue, source)
  ╰─ target[tok] := with_head_tail(<s>, </s>)(target.tok)
  ╰─ target[trunc_tok] := trunc_and_pad(100, </unk>)(target.tok)
  ╰─ target[trunc_len] := nestedmaxlength(target.trunc_tok)
  ╰─ target := (target.trunc_tok, target.trunc_len)
)

```
"""
struct TransformerTextEncoder{T<:AbstractTokenizer, V<:AbstractVocabulary{String}, P} <: AbstractTextEncoder
    tokenizer::T
    vocab::V
    process::P
    startsym::String
    endsym::String
    padsym::String
    trunc::Union{Nothing, Int}
end

# encoder constructor

const WList = Union{AbstractVector, AbstractVocabulary}

TransformerTextEncoder(tokenizef, v::WList, args...; kws...) =
    TransformerTextEncoder(WordTokenization(tokenize=tokenizef), v, args...; kws...)
TransformerTextEncoder(v::WList, args...; kws...) = TransformerTextEncoder(TextTokenizer(), v, args...; kws...)
TransformerTextEncoder(t::AbstractTokenization, v::WList, args...; kws...) =
    TransformerTextEncoder(TextTokenizer(t), v, args...; kws...)

function TransformerTextEncoder(tkr::AbstractTokenizer, words::AbstractVector, process; trunc = nothing,
                                startsym = "<s>", endsym = "</s>", unksym = "<unk>", padsym = "<pad>")
    vocab_list = String[]
    for sym in (padsym, unksym, startsym, endsym)
        sym ∉ words && push!(vocab_list, sym)
    end
    append!(vocab_list, words)
    vocab = Vocab(vocab_list, unksym)
    return TransformerTextEncoder(tkr, vocab, process, startsym, endsym, padsym, trunc)
end

function TransformerTextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary, process; trunc = nothing,
                                startsym = "<s>", endsym = "</s>", unksym = "<unk>", padsym = "<pad>")
    check_vocab(vocab, startsym) || @warn "startsym $startsym not in vocabulary, this might cause problem."
    check_vocab(vocab, endsym) || @warn "endsym $endsym not in vocabulary, this might cause problem."
    check_vocab(vocab, unksym) || @warn "unksym $unksym not in vocabulary, this might cause problem."
    check_vocab(vocab, padsym) || @warn "padsym $padsym not in vocabulary, this might cause problem."
    return TransformerTextEncoder(tkr, vocab, process, startsym, endsym, padsym, trunc)
end

function TransformerTextEncoder(tkr::AbstractTokenizer, v::WList; kws...)
    enc = TransformerTextEncoder(tkr, v, identity; kws...)
    # default processing pipeline
    return TransformerTextEncoder(enc) do e
        # get token and convert to string
        Pipeline{:tok}(nestedcall(string_getvalue), 1) |>
            # add start & end symbol
            Pipeline{:tok}(with_head_tail(e.startsym, e.endsym), :tok) |>
            # truncate input that exceed length limit and pad them to have equal length
            Pipeline{:trunc_tok}(trunc_and_pad(e.trunc, e.padsym), :tok) |>
            # get the truncated length
            Pipeline{:trunc_len}(TextEncodeBase.nestedmaxlength, :trunc_tok) |>
            # get mask with specific length
            Pipeline{:mask}(getmask, (:tok, :trunc_len)) |>
            # convert to dense array
            Pipeline{:tok}(nested2batch, :trunc_tok) |>
            # return token and mask
            PipeGet{(:tok, :mask)}()
    end
end

TransformerTextEncoder(builder, e::TransformerTextEncoder) = TransformerTextEncoder(
    e.tokenizer, e.vocab, builder(e), e.startsym, e.endsym, e.padsym, e.trunc)

# encoder behavior

TextEncodeBase.tokenize(e::TransformerTextEncoder, x::AbstractString) = e.tokenizer(Sentence(x))
TextEncodeBase.tokenize(e::TransformerTextEncoder, x::Vector{<:AbstractString}) =
    e.tokenizer(Batch{Sentence}(x))

TextEncodeBase.lookup(e::TransformerTextEncoder, x::Tuple) = (lookup(e, x[1]), Base.tail(x)...)
function TextEncodeBase.lookup(e::TransformerTextEncoder, x::NamedTuple{name}) where name
    xt = Tuple(x)
    return NamedTuple{name}((lookup(e, xt[1]), Base.tail(xt)...))
end

# pretty print

function Base.show(io::IO, e::TransformerTextEncoder)
    print(io, "TransformerTextEncoder(\n├─ ")
    print(io, e.tokenizer, ",\n├─ ")
    print(io, "vocab = ", e.vocab, ",\n├─ ")
    print(io, "startsym = ", e.startsym, ",\n├─ ")
    print(io, "endsym = ", e.endsym, ",\n├─ ")
    print(io, "padsym = ", e.padsym)
    isnothing(e.trunc) || print(io, ",\n├─ trunc = ", e.trunc)
    print(IOContext(io, :pipeline_display_prefix => "  ╰─ "), ",\n└─ process = ", e.process, "\n)")
end
