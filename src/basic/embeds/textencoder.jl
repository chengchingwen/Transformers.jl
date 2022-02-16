using TextEncodeBase: trunc_and_pad, nested2batch, with_head_tail, nestedcall, getvalue

struct TransformerTextEncoder{T<:AbstractTokenizer, V<:AbstractVocabulary{String}} <: AbstractTextEncoder
    tkr::T
    vocab::V
    startsym::String
    endsym::String
    padsym::String
    trunc::Union{Nothing, Int}
end

TransformerTextEncoder(words::AbstractVector; kws...) = TransformerTextEncoder(TextEncodeBase.NestedTokenizer(), words; kws...)
function TransformerTextEncoder(tkr::AbstractTokenizer, words::AbstractVector; trunc = nothing,
                                startsym = "<s>", endsym = "</s>", unksym = "<unk>", padsym = "<pad>")
    vocab_list = append!([padsym, unksym, startsym, endsym], words)
    vocab = Vocab(vocab_list, unksym)
    return TransformerTextEncoder(tkr, vocab, startsym, endsym, padsym, trunc)
end

TextEncodeBase.tokenize(e::TransformerTextEncoder, x::String) = e.tkr(TextEncodeBase.Document(x))
TextEncodeBase.tokenize(e::TransformerTextEncoder, x) = e.tkr(x)

TextEncodeBase.process(e::TransformerTextEncoder, x) = nested2batch(
    trunc_and_pad(with_head_tail(nestedcall(getvalue, x), e.startsym, e.endsym), e.trunc, e.padsym)
)

