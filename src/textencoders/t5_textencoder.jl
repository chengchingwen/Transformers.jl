using TextEncodeBase
using TextEncodeBase: trunc_and_pad, trunc_or_pad, nested2batch, nestedcall, Batch, Sentence
using TextEncodeBase: SequenceTemplate, ConstTerm, InputTerm, RepeatedTerm
using FuncPipelines

struct T5TextEncoder{T <: AbstractTokenizer, V <: AbstractVocabulary{String}, P} <: AbstractTransformerTextEncoder
    tokenizer::T
    vocab::V
    process::P
    endsym::String
    padsym::String
    trunc::Union{Nothing, Int}
end

function T5TextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary, process;
              endsym = "</s>", padsym = "<pad>", trunc = nothing)
    check_vocab(vocab, endsym) || @warn "endsym $endsym not in vocabulary, this might cause problem."
    check_vocab(vocab, padsym) || @warn "padsym $padsym not in vocabulary, this might cause problem."
    return T5TextEncoder(tkr, vocab, process, endsym, padsym, trunc)
end

function T5TextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary;
                       fixedsize = false, trunc_end = :tail, pad_end = :tail, process = nothing,
                       kws...)
    enc = T5TextEncoder(tkr, vocab, TextEncodeBase.process(AbstractTextEncoder); kws...)
    return T5TextEncoder(enc) do e
        t5_default_preprocess(; trunc = e.trunc, endsym = e.endsym, padsym = e.padsym,
                              fixedsize, trunc_end, pad_end, process)
    end
end

T5TextEncoder(builder, e::T5TextEncoder) =
    T5TextEncoder(e.tokenizer, e.vocab, builder(e), e.endsym, e.padsym, e.trunc)

function t5_default_preprocess(; startsym = "[CLS]", endsym = "[SEP]", padsym = "[PAD]",
                                 fixedsize = false, trunc = nothing, trunc_end = :tail, pad_end = :tail,
                                 process = nothing)
    truncf = get_trunc_pad_func(padsym, fixedsize, trunc, trunc_end, pad_end)
    maskf = get_mask_func(trunc, pad_end)
    if isnothing(process)
        process =
            # group input for SequenceTemplate
            Pipeline{:token}(grouping_sentence, :token) |>
            # add start & end symbol, compute segment and merge sentences
            Pipeline{:token}(
                SequenceTemplate(
                    InputTerm{String}(), ConstTerm(endsym),
                    RepeatedTerm(InputTerm{String}(), ConstTerm(endsym, 2)))(Val(1)), :token)
    end

    # get token and convert to string
    return Pipeline{:token}(nestedcall(string_getvalue), 1) |>
        process |>
        # get mask with specific length
        Pipeline{:attention_mask}(maskf, :token) |>
        # truncate input that exceed length limit and pad them to have equal length
        Pipeline{:token}(truncf, :token) |>
        # convert to dense array
        Pipeline{:token}(nested2batch, :token) |>
        # return input and mask
        PipeGet{(:token, :attention_mask)}()
end
