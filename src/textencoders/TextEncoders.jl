module TextEncoders

using TextEncodeBase
using TextEncodeBase: WordTokenization, nested2batch, nestedcall, with_head_tail, tokenize
using ..WordPieceModel
using BytePairEncoding
using ..UnigramLanguageModel

using NeuralAttentionlib: AttenMask, LengthMask, RevLengthMask, GenericSequenceMask

export lookup, encode, decode, Vocab, OneHot,
    TransformerTextEncoder, BertTextEncoder, GPT2TextEncoder, T5TextEncoder



include("bert_tokenizer.jl")
include("gpt_tokenizer.jl")

include("utils.jl")
include("tokenizer.jl")

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
        truncf = get_trunc_pad_func(e.padsym, false, e.trunc, :tail, :tail)
        maskf = get_mask_func(e.trunc, :tail)
        # get token and convert to string
        Pipeline{:token}(nestedcall(string_getvalue), 1) |>
            # add start & end symbol
            Pipeline{:token}(with_head_tail(e.startsym, e.endsym), :token) |>
            # get mask with specific length
            Pipeline{:attention_mask}(maskf, :token) |>
            # truncate input that exceed length limit and pad them to have equal length
            Pipeline{:token}(truncf, :token) |>
            # convert to dense array
            Pipeline{:token}(nested2batch, :token) |>
            # return token and mask
            PipeGet{(:token, :attention_mask)}()
    end
end

TransformerTextEncoder(builder, e::TransformerTextEncoder) = TransformerTextEncoder(
    e.tokenizer, e.vocab, builder(e), e.startsym, e.endsym, e.padsym, e.trunc)

# encoder behavior

TextEncodeBase.tokenize(e::AbstractTransformerTextEncoder, x::AbstractString) = e.tokenizer(Sentence(x))
TextEncodeBase.tokenize(e::AbstractTransformerTextEncoder, x::Vector{<:AbstractString}) = e.tokenizer(Batch{Sentence}(x))
TextEncodeBase.tokenize(e::AbstractTransformerTextEncoder, x::Vector{<:Vector{<:AbstractString}}) =
    e.tokenizer(Batch{Batch{Sentence}}(x))
TextEncodeBase.tokenize(e::AbstractTransformerTextEncoder, x::Vector{<:Vector{<:Vector{<:AbstractString}}}) =
    e.tokenizer(Batch{Batch{Batch{Sentence}}}(x))

TextEncodeBase.lookup(e::AbstractTransformerTextEncoder, x::Tuple) = (lookup(e, x[1]), Base.tail(x)...)
function TextEncodeBase.lookup(e::AbstractTransformerTextEncoder, x::NamedTuple{name}) where name
    xt = Tuple(x)
    return NamedTuple{name}((lookup(e, xt[1]), Base.tail(xt)...))
end

## encoder-decoder encoding
function TextEncodeBase.encode(e::AbstractTransformerTextEncoder, src, trg)
    psrc = encode(e, src)
    ptrg = encode(e, trg)
    cross_attention_mask = AttenMask(ptrg.attention_mask, psrc.attention_mask)
    return (encoder_input = psrc, decoder_input = merge(ptrg, (; cross_attention_mask)))
end

include("bert_textencoder.jl")
include("gpt_textencoder.jl")
include("t5_textencoder.jl")

end
