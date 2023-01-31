module TextEncoders

using PrimitiveOneHot
using FuncPipelines
using TextEncodeBase
using TextEncodeBase: WordTokenization, nested2batch, nestedcall, with_head_tail, tokenize
using ..WordPieceModel
using BytePairEncoding
using ..UnigramLanguageModel

using NeuralAttentionlib: AttenMask, LengthMask, RevLengthMask, GenericSequenceMask

export lookup, encode, decode, Vocab, OneHot, OneHotArray,
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

struct TransformerTextEncoder{
    T <: AbstractTokenizer, V <: AbstractVocabulary{String}, P
} <: AbstractTransformerTextEncoder
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

# decoder behavior
function TextEncodeBase.decode(e::AbstractTransformerTextEncoder,
                               i::Union{Integer, OneHotArray, AbstractArray{<:Integer}})
    return TextEncodeBase.decode_indices(e, i)
end

TextEncodeBase.decode(e::AbstractTransformerTextEncoder, x::AbstractVector) = decode(e, argmax(x))
function TextEncodeBase.decode(e::AbstractTransformerTextEncoder, x::AbstractArray)
    amax = reshape(argmax(x; dims=1), Base.tail(size(x)))
    i = selectdim(reinterpret(reshape, Int, amax), 1, 1)
    return decode(e, i)
end


include("bert_textencoder.jl")
include("gpt_textencoder.jl")
include("t5_textencoder.jl")


"""
    struct TransformerTextEncoder{
        T<:AbstractTokenizer, V<:AbstractVocabulary{String}, P
    } <: AbstractTransformerTextEncoder
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
 `process` can be omitted, then a predefined processing pipeline will be used. When `vocab` is a list, those
 special symbol (e.g. `padsym`) would be added to the word list.

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
  ╰─ target[token] := TextEncodeBase.nestedcall(string_getvalue, source)
  ╰─ target[token] := TextEncodeBase.with_head_tail(<s>, </s>)(target.token)
  ╰─ target[attention_mask] := (NeuralAttentionlib.LengthMask ∘ Transformers.TextEncoders.getlengths(10))(target.token)
  ╰─ target[token] := TextEncodeBase.trunc_and_pad(10, <pad>, tail, tail)(target.token)
  ╰─ target[token] := TextEncodeBase.nested2batch(target.token)
  ╰─ target := (target.token, target.attention_mask)
)

julia> TransformerTextEncoder(ans) do enc
           enc.process[1] |> TextEncoders.Pipelines(enc.process[4:5]) |> TextEncoders.PipeGet{(:token,)}()
       end
TransformerTextEncoder(
├─ TextTokenizer(default),
├─ vocab = Vocab{String, SizedArray}(size = 37678, unk = </unk>, unki = 1),
├─ startsym = <s>,
├─ endsym = </s>,
├─ padsym = </unk>,
├─ trunc = 100,
└─ process = Pipelines:
  ╰─ target[token] := TextEncodeBase.nestedcall(string_getvalue, source)
  ╰─ target[token] := TextEncodeBase.trunc_and_pad(10, <pad>, tail, tail)(target.token)
  ╰─ target[token] := TextEncodeBase.nested2batch(target.token)
  ╰─ target := (target.token)
)

```
"""
TransformerTextEncoder

"""
    encode(e::AbstractTransformerTextEncoder, input::Union{
        String,                         # single sentence
        Vector{String},                 # batch of sentences
        Vector{Vector{String}},         # batch of multi-segment sentences
        Vector{Vector{Vector{String}}}  # batch of multi-sample multi-segment sentences
    })

Tokenize the `input` and apply the processing function on the tokenized result. The `input` can be either a single
 `String` (1 sample) or a nested vector of `String` up to depth 3 (batch of samples). How batch input is transformed
 is defined by the bound processing function. The result of the processing function (first if return tuple) would be
 converted into one-hot encoding with the bound vocabulary.

    encode(e::AbstractTransformerTextEncoder, src, trg)

Apply `encode` on `src` and `trg` and build the cross attention mask. This is just a convenient function for doing
 encoder-decoder tasks. Return a `@NamedTuple{encoder_input, decoder_input}` where `encoder_input` is just
 `encode(e, src)` and `decoder_input` is `encode(e, trg)` + the cross attention mask.
"""
TextEncodeBase.encode(e::AbstractTransformerTextEncoder, x)

"""
    decode(e::AbstractTransformerTextEncoder, x::Union{
        Integer,
        OneHotArray,
        AbstractArray{<:Integer}
    })

Decode the one-hot encoding or indices into `String` (or `Array{String}`) from the bound vocabulary.

    decode(e::AbstractTransformerTextEncoder, x::AbstractArray)

Perform `argmax(x; dims = 1)` and then `decode`. `x` should be `collect`ed beforehand if it's on GPU.
"""
TextEncodeBase.decode(e::AbstractTransformerTextEncoder, x)

end
