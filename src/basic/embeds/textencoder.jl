using InternedStrings
using TextEncodeBase: trunc_and_pad, nested2batch, with_head_tail, nestedcall, getvalue
using TextEncodeBase: WordTokenization

string_getvalue(x::TextEncodeBase.TokenStage) = intern(getvalue(x))::String

# abstract type

abstract type AbstractTransformerTextEncoder <: AbstractTextEncoder end

function Base.show(io::IO, e::AbstractTransformerTextEncoder)
    print(io, "$(nameof(typeof(e)))(\n├─ ")
    print(io, e.tokenizer)
    for name in fieldnames(typeof(e))
        (name == :tokenizer || name == :process) && continue
        val = getfield(e, name)
        isnothing(val) || print(io, ",\n├─ ", name, " = ",  val)
    end
    print(IOContext(io, :pipeline_display_prefix => "  ╰─ "), ",\n└─ process = ", e.process, "\n)")
end

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
struct TransformerTextEncoder{T <: AbstractTokenizer, V <: AbstractVocabulary{String}, P} <: AbstractTransformerTextEncoder
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
    vocab_list = copy(words)
    for sym in (padsym, unksym, startsym, endsym)
        sym ∉ vocab_list && push!(vocab_list, sym)
    end
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

TransformerTextEncoder(builder, e::AbstractTransformerTextEncoder) = TransformerTextEncoder(
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

# api doc

"""
    encode(::TransformerTextEncoder, ::String)

Encode a single sentence with bert text encoder. The default pipeline returning
 `@NamedTuple{tok::OneHotArray{K, 2}, mask::Nothing}`

    encode(::TransformerTextEncoder, ::Vector{String})

Encode a batch of sentences with bert text encoder. The default pipeline returning
 `@NamedTuple{tok::OneHotArray{K, 3}, mask::Array{Float32, 3}}`

See also: [`decode`](@ref)

# Example
```julia-repl
julia> textenc = TransformerTextEncoder(split, map(string, 1:10))
TransformerTextEncoder(
├─ TextTokenizer(WordTokenization(split_sentences = WordTokenizers.split_sentences, tokenize = split)),
├─ vocab = Vocab{String, SizedArray}(size = 14, unk = <unk>, unki = 2),
├─ startsym = <s>,
├─ endsym = </s>,
├─ padsym = <pad>,
└─ process = Pipelines:
  ╰─ target[tok] := nestedcall(string_getvalue, source)
  ╰─ target[tok] := with_head_tail(<s>, </s>)(target.tok)
  ╰─ target[trunc_tok] := trunc_and_pad(nothing, <pad>)(target.tok)
  ╰─ target[trunc_len] := nestedmaxlength(target.trunc_tok)
  ╰─ target[mask] := getmask(target.tok, target.trunc_len)
  ╰─ target[tok] := nested2batch(target.trunc_tok)
  ╰─ target := (target.tok, target.mask)
)

julia> e = encode(textenc, ["1 2 3 4 5 6 7", join(rand(1:10 , 9), ' ')])
(tok = [0 0 … 1 1; 0 0 … 0 0; … ; 0 0 … 0 0; 0 0 … 0 0;;; 0 0 … 0 0; 0 0 … 0 0; … ; 0 0 … 0 0; 0 0 … 0 0], mask = [1.0 1.0 … 0.0 0.0;;; 1.0 1.0 … 1.0 1.0])

julia> typeof(e)
NamedTuple{(:tok, :mask), Tuple{OneHotArray{0x0000000e, 2, 3, Matrix{OneHot{0x0000000e}}}, Array{Float32, 3}}}

```
"""
TextEncodeBase.encode(::TransformerTextEncoder, _)

"""
    decode(textenc::TransformerTextEncoder, x)

Equivalent to `lookup(textenc.vocab, x)`.

See also: [`encode`](@ref)

# Example
```julia-repl
julia> textenc = TransformerTextEncoder(split, map(string, 1:10));

julia> e = encode(textenc, ["1 2 3 4 5 6 7", join(rand(1:10 , 9), ' ')]);

julia> decode(textenc, e.tok)
11×2 Matrix{String}:
 "<s>"    "<s>"
 "1"      "3"
 "2"      "5"
 "3"      "4"
 "4"      "6"
 "5"      "6"
 "6"      "5"
 "7"      "5"
 "</s>"   "6"
 "<pad>"  "6"
 "<pad>"  "</s>"

julia> lookup(textenc.vocab, e.tok)
11×2 Matrix{String}:
 "<s>"    "<s>"
 "1"      "3"
 "2"      "5"
 "3"      "4"
 "4"      "6"
 "5"      "6"
 "6"      "5"
 "7"      "5"
 "</s>"   "6"
 "<pad>"  "6"
 "<pad>"  "</s>"

```
"""
TextEncodeBase.decode(::TransformerTextEncoder, _)
