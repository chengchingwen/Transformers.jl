using TextEncodeBase: trunc_and_pad, nested2batch, with_head_tail, nestedcall, getvalue

struct TransformerTextEncoder{T<:AbstractTokenizer, V<:AbstractVocabulary{String}, P, N<:Union{Nothing, Int}} <: AbstractTextEncoder
    tokenizer::T
    vocab::V
    process::P
    startsym::String
    endsym::String
    padsym::String
    trunc::N
end

string_getvalue(x::TextEncodeBase.TokenStage) = getvalue(x)::String
_get_tok(y) = y.tok

TransformerTextEncoder(words::AbstractVector; kws...) = TransformerTextEncoder(TextTokenizer(), words; kws...)
function TransformerTextEncoder(tkr::AbstractTokenizer, words::AbstractVector; kws...)
    enc = TransformerTextEncoder(tkr, words, TextEncodeBase.process(AbstractTextEncoder); kws...)
    return TransformerTextEncoder(enc) do e
        Pipeline{:tok}(nestedcall(string_getvalue), 1) |>
            Pipeline{:tok}(with_head_tail(e.startsym, e.endsym), :tok) |>
            Pipeline{:mask}(getmask, :tok) |>
            Pipeline{:tok}(nested2batch∘trunc_and_pad(e.trunc, e.padsym), :tok)
    end
end

TransformerTextEncoder(words::AbstractVector, process; kws...) = TransformerTextEncoder(TextTokenizer(), words, process; kws...)
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
    return TransformerTextEncoder(tkr, vocab, process, startsym, endsym, padsym, trunc)
end

TransformerTextEncoder(builder::Base.Callable, args...; kwargs...) =
    TransformerTextEncoder(builder, TransformerTextEncoder(args...; kwargs...))

TransformerTextEncoder(builder::Base.Callable, e::TransformerTextEncoder) = TransformerTextEncoder(
    e.tokenizer,
    e.vocab,
    builder(e),
    e.startsym,
    e.endsym,
    e.padsym,
    e.trunc
)

TransformerTextEncoder(builder, e::TransformerTextEncoder) = TransformerTextEncoder(
    e.tokenizer,
    e.vocab,
    builder(e),
    e.startsym,
    e.endsym,
    e.padsym,
    e.trunc
)

TextEncodeBase.tokenize(e::TransformerTextEncoder, x::AbstractString) = e.tokenizer(Sentence(x))
TextEncodeBase.tokenize(e::TransformerTextEncoder, x::Vector{<:AbstractString}) =
    e.tokenizer(Batch{Sentence}(x))

TextEncodeBase.lookup(e::TransformerTextEncoder, x::Tuple) = (lookup(e, x[1]), Base.tail(x)...)
function TextEncodeBase.lookup(e::TransformerTextEncoder, x::NamedTuple{name}) where name
    xt = Tuple(x)
    return NamedTuple{name}((lookup(e, xt[1]), Base.tail(xt)...))
end


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
