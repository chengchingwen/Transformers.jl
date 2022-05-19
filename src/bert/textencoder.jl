using ..Transformers: Container
using ..Basic: string_getvalue, _get_tok, TextTokenizer
using TextEncodeBase
using TextEncodeBase: trunc_and_pad, nested2batch, nestedcall
using TextEncodeBase: BaseTokenization, WrappedTokenization, Splittable,
    ParentStages, TokenStages, SentenceStage, WordStage, Batch, Sentence, getvalue, getmeta

struct BertCasedPreTokenization   <: BaseTokenization end
struct BertUnCasedPreTokenization <: BaseTokenization end

TextEncodeBase.splitting(::BertCasedPreTokenization, s::SentenceStage) = bert_cased_tokenizer(getvalue(s))
TextEncodeBase.splitting(::BertUnCasedPreTokenization, s::SentenceStage) = bert_uncased_tokenizer(getvalue(s))

struct WordPieceTokenization{T<:AbstractTokenization} <: WrappedTokenization{T}
    base::T
    wordpiece::WordPiece
end

TextEncodeBase.splittability(::ParentStages, ::WordPieceTokenization, ::WordStage) = Splittable()
TextEncodeBase.splitting(::ParentStages, t::WordPieceTokenization, w::WordStage) = t.wordpiece(getvalue(w))

struct BertTextEncoder{T<:AbstractTokenizer, V<:AbstractVocabulary{String}, P, N<:Union{Nothing, Int}} <: AbstractTextEncoder
    tokenizer::T
    vocab::V
    process::P
    startsym::String
    endsym::String
    trunc::N
end

BertTextEncoder(t::WordPieceTokenization; kws...) = BertTextEncoder(TextTokenizer(t), t.wordpiece; kws...)
BertTextEncoder(tkr::AbstractTokenizer, wordpiece::WordPiece; kws...) = BertTextEncoder(tkr, Vocab(wordpiece); kws...)
function BertTextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary; kws...)
    enc = BertTextEncoder(tkr, vocab, TextEncodeBase.process(AbstractTextEncoder); kws...)
    return BertTextEncoder(enc) do e
        Pipeline{:tok}(nestedcall(string_getvalue), 1) |>
            Pipeline{:tok}(with_firsthead_tail(e.startsym, e.endsym), :tok) |>
            Pipeline{(:tok, :segment)}(segment_and_concat, :tok) |>
            Pipeline{:mask}(getmask, :tok) |>
            Pipeline{:tok}(nested2batch∘trunc_and_pad(e.trunc, e.vocab.unk), :tok) |>
            Pipeline{:segment}(nested2batch∘trunc_and_pad(e.trunc, 1), :segment) |>
            Pipeline{:input}(build_input) |>
            PipeGet{(:input, :mask)}()
    end
end

BertTextEncoder(::typeof(bert_cased_tokenizer), wordpiece::WordPiece, process; kws...) = BertTextEncoder(TextTokenizer(WordPieceTokenization(bert_cased_tokenizer, wordpiece)), wordpiece, process; kws...)
BertTextEncoder(::typeof(bert_uncased_tokenizer), wordpiece::WordPiece, process; kws...) = BertTextEncoder(TextTokenizer(WordPieceTokenization(bert_uncased_tokenizer, wordpiece)), wordpiece, process; kws...)

BertTextEncoder(t::WordPieceTokenization, process; kws...) = BertTextEncoder(TextTokenizer(t), t.wordpiece, process; kws...)
BertTextEncoder(tkr::AbstractTokenizer, wordpiece::WordPiece, process; kws...) = BertTextEncoder(tkr, Vocab(wordpiece), process; kws...)
function BertTextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary, process; startsym = "[CLS]", endsym = "[SEP]", trunc = nothing)
    return BertTextEncoder(tkr, vocab, process, startsym, endsym, trunc)
end

BertTextEncoder(builder, e::BertTextEncoder) = BertTextEncoder(
    e.tokenizer,
    e.vocab,
    builder(e),
    e.startsym,
    e.endsym,
    e.trunc
)


TextEncodeBase.tokenize(e::BertTextEncoder, x::AbstractString) = e.tokenizer(Sentence(x))
TextEncodeBase.tokenize(e::BertTextEncoder, x::Vector{<:AbstractString}) = e.tokenizer(Batch{Sentence}(x))
TextEncodeBase.tokenize(e::BertTextEncoder, x::Vector{<:Container{<:AbstractString}}) = e.tokenizer(Batch{Batch{Sentence}}(x))

function TextEncodeBase.lookup(e::BertTextEncoder, x::NamedTuple)
    onehot_tok = lookup(e.vocab, x.input.tok)
    input = merge(x.input, (tok = onehot_tok,))
    return merge(x, (input = input,))
end

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
