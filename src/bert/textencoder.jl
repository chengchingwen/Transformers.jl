using ..Basic: string_getvalue, check_vocab, TextTokenizer
using TextEncodeBase
using TextEncodeBase: trunc_and_pad, nested2batch, nestedcall
using TextEncodeBase: BaseTokenization, WrappedTokenization, Splittable,
    ParentStages, TokenStages, SentenceStage, WordStage, Batch, Sentence, getvalue, getmeta

# bert tokenizer

struct BertCasedPreTokenization   <: BaseTokenization end
struct BertUnCasedPreTokenization <: BaseTokenization end

TextEncodeBase.splitting(::BertCasedPreTokenization, s::SentenceStage) = bert_cased_tokenizer(getvalue(s))
TextEncodeBase.splitting(::BertUnCasedPreTokenization, s::SentenceStage) = bert_uncased_tokenizer(getvalue(s))

const BertTokenization = Union{BertCasedPreTokenization, BertUnCasedPreTokenization}

struct WordPieceTokenization{T<:AbstractTokenization} <: WrappedTokenization{T}
    base::T
    wordpiece::WordPiece
end

TextEncodeBase.splittability(::ParentStages, ::WordPieceTokenization, ::WordStage) = Splittable()
TextEncodeBase.splitting(::ParentStages, t::WordPieceTokenization, w::WordStage) = t.wordpiece(getvalue(w))

# encoder

"""
    struct BertTextEncoder{T<:AbstractTokenizer,
                           V<:AbstractVocabulary{String},
                           P} <: AbstractTextEncoder
      tokenizer::T
      vocab::V
      process::P
      startsym::String
      endsym::String
      trunc::Union{Nothing, Int}
    end

The text encoder for Bert model. Taking a tokenizer, vocabulary, and a processing function, configured with
 a start symbol, an end symbol, and a maximum length.

    BertTextEncoder(bert_tokenizer, wordpiece, process;
                    startsym = "[CLS]", endsym = "[SEP]", trunc = nothing)

There are two tokenizer supported (`bert_cased_tokenizer` and `bert_uncased_tokenizer`).
 `process` can be omitted, then a predefined processing pipeline will be used.


    BertTextEncoder(f, bertenc::BertTextEncoder)

Take a bert text encoder and create a new bert text encoder with same configuration except the processing function.
 `f` is a function that take the encoder and return a new process function. This is useful for changing part of
 the procssing function.

# Example

```julia-repl
julia> wordpiece = pretrain"bert-cased_L-12_H-768_A-12:wordpiece"
[ Info: loading pretrain bert model: cased_L-12_H-768_A-12.tfbson wordpiece
WordPiece(vocab_size=28996, unk=[UNK], max_char=200)

julia> bertenc = BertTextEncoder(bert_cased_tokenizer, wordpiece; trunc=5)
BertTextEncoder(
├─ TextTokenizer(WordPieceTokenization(bert_cased_tokenizer, WordPiece(vocab_size=28996, unk=[UNK], max_char=200))),
├─ vocab = Vocab{String, SizedArray}(size = 28996, unk = [UNK], unki = 101),
├─ startsym = [CLS],
├─ endsym = [SEP],
├─ trunc = 5,
└─ process = Pipelines:
  ╰─ target[tok] := nestedcall(string_getvalue, source)
  ╰─ target[tok] := with_firsthead_tail([CLS], [SEP])(target.tok)
  ╰─ target[(tok, segment)] := segment_and_concat(target.tok)
  ╰─ target[trunc_tok] := trunc_and_pad(nothing, [UNK])(target.tok)
  ╰─ target[trunc_len] := nestedmaxlength(target.trunc_tok)
  ╰─ target[mask] := getmask(target.tok, target.trunc_len)
  ╰─ target[tok] := nested2batch(target.trunc_tok)
  ╰─ target[segment] := (nested2batch ∘ trunc_and_pad(nothing, 1))(target.segment)
  ╰─ target[input] := (NamedTuple{(:tok, :segment)} ∘ tuple)(target.tok, target.segment)
  ╰─ target := (target.input, target.mask)
)

# take the first 3 pipeline and get the result
julia> BertTextEncoder(bertenc) do enc
           Pipelines(enc.process[1:3]) |> PipeGet{(:tok, :segment)}()
       end
BertTextEncoder(
├─ TextTokenizer(WordPieceTokenization(bert_cased_tokenizer, WordPiece(vocab_size=28996, unk=[UNK], max_char=200))),
├─ vocab = Vocab{String, SizedArray}(size = 28996, unk = [UNK], unki = 101),
├─ startsym = [CLS],
├─ endsym = [SEP],
└─ process = Pipelines:
  ╰─ target[tok] := nestedcall(string_getvalue, source)
  ╰─ target[tok] := with_firsthead_tail([CLS], [SEP])(target.tok)
  ╰─ target[(tok, segment)] := segment_and_concat(target.tok)
  ╰─ target := (target.tok, target.segment)
)

```
"""
struct BertTextEncoder{T<:AbstractTokenizer,
                       V<:AbstractVocabulary{String},
                       P} <: AbstractTextEncoder
    tokenizer::T
    vocab::V
    process::P
    startsym::String
    endsym::String
    trunc::Union{Nothing, Int}
end

# encoder constructor

BertTextEncoder(::typeof(bert_cased_tokenizer), args...; kws...) =
    BertTextEncoder(BertCasedPreTokenization(), args...; kws...)
BertTextEncoder(::typeof(bert_uncased_tokenizer), args...; kws...) =
    BertTextEncoder(BertUnCasedPreTokenization(), args...; kws...)
BertTextEncoder(bt::BertTokenization, wordpiece::WordPiece, args...; kws...) =
    BertTextEncoder(WordPieceTokenization(bt, wordpiece), args...; kws...)
BertTextEncoder(t::WordPieceTokenization, args...; kws...) =
    BertTextEncoder(TextTokenizer(t), Vocab(t.wordpiece), args...; kws...)
BertTextEncoder(t::AbstractTokenization, vocab::AbstractVocabulary, args...; kws...) =
    BertTextEncoder(TextTokenizer(t), vocab, args...; kws...)

function BertTextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary, process;
                         startsym = "[CLS]", endsym = "[SEP]", trunc = nothing)
    check_vocab(vocab, startsym) || @warn "startsym $startsym not in vocabulary, this might cause problem."
    check_vocab(vocab, endsym) || @warn "endsym $endsym not in vocabulary, this might cause problem."
    return BertTextEncoder(tkr, vocab, process, startsym, endsym, trunc)
end

function BertTextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary; kws...)
    enc = BertTextEncoder(tkr, vocab, TextEncodeBase.process(AbstractTextEncoder); kws...)
    # default processing pipelines for bert encoder
    return BertTextEncoder(enc) do e
        # get token and convert to string
        Pipeline{:tok}(nestedcall(string_getvalue), 1) |>
            # add start & end symbol
            Pipeline{:tok}(with_firsthead_tail(e.startsym, e.endsym), :tok) |>
            # compute segment and merge sentences
            Pipeline{(:tok, :segment)}(segment_and_concat, :tok) |>
            # truncate input that exceed length limit and pad them to have equal length
            Pipeline{:trunc_tok}(trunc_and_pad(e.trunc, e.vocab.unk), :tok) |>
            # get the truncated length
            Pipeline{:trunc_len}(TextEncodeBase.nestedmaxlength, :trunc_tok) |>
            # get mask with specific length
            Pipeline{:mask}(getmask, (:tok, :trunc_len)) |>
            # convert to dense array
            Pipeline{:tok}(nested2batch, :trunc_tok) |>
            # truncate & pad segment
            Pipeline{:segment}(nested2batch∘trunc_and_pad(e.trunc, 1), :segment) |>
            # input namedtuple
            Pipeline{:input}(NamedTuple{(:tok, :segment)}∘tuple, (:tok, :segment)) |>
            # return input and mask
            PipeGet{(:input, :mask)}()
    end
end

BertTextEncoder(builder, e::BertTextEncoder) =
    BertTextEncoder(e.tokenizer, e.vocab, builder(e), e.startsym, e.endsym, e.trunc)

# encoder behavior

TextEncodeBase.tokenize(e::BertTextEncoder, x::AbstractString) = e.tokenizer(Sentence(x))
TextEncodeBase.tokenize(e::BertTextEncoder, x::Vector{<:AbstractString}) = e.tokenizer(Batch{Sentence}(x))
TextEncodeBase.tokenize(e::BertTextEncoder, x::Vector{<:Vector{<:AbstractString}}) = e.tokenizer(Batch{Batch{Sentence}}(x))

function TextEncodeBase.lookup(e::BertTextEncoder, x::NamedTuple)
    onehot_tok = lookup(e.vocab, x.input.tok)
    input = merge(x.input, (tok = onehot_tok,))
    return merge(x, (input = input,))
end

# pretty print

function Base.show(io::IO, e::BertTextEncoder)
    print(io, "BertTextEncoder(\n├─ ")
    print(io, e.tokenizer, ",\n├─ ")
    print(io, "vocab = ", e.vocab, ",\n├─ ")
    print(io, "startsym = ", e.startsym, ",\n├─ ")
    print(io, "endsym = ", e.endsym)
    isnothing(e.trunc) || print(io, ",\n├─ trunc = ", e.trunc)
    print(IOContext(io, :pipeline_display_prefix => "  ╰─ "), ",\n└─ process = ", e.process, "\n)")
end

Base.show(io::IO, ::BertCasedPreTokenization) = print(io, nameof(bert_cased_tokenizer))
Base.show(io::IO, ::BertUnCasedPreTokenization) = print(io, nameof(bert_uncased_tokenizer))
Base.show(io::IO, wp::WordPieceTokenization) = print(io, "WordPieceTokenization(", wp.base, ", ", wp.wordpiece, ')')
