using ..WordPieceModel
using ..WordPieceModel: DAT
using FuncPipelines
using TextEncodeBase
using TextEncodeBase: nested2batch, nestedcall
using TextEncodeBase: BaseTokenization, WrappedTokenization, MatchTokenization, Splittable,
    ParentStages, TokenStages, SentenceStage, WordStage, Batch, Sentence, getvalue, getmeta
using TextEncodeBase: SequenceTemplate, ConstTerm, InputTerm, RepeatedTerm

# bert tokenizer

struct BertCasedPreTokenization   <: BaseTokenization end
struct BertUnCasedPreTokenization <: BaseTokenization end

TextEncodeBase.splitting(::BertCasedPreTokenization, s::SentenceStage) = bert_cased_tokenizer(getvalue(s))
TextEncodeBase.splitting(::BertUnCasedPreTokenization, s::SentenceStage) = bert_uncased_tokenizer(getvalue(s))

const BertTokenization = Union{BertCasedPreTokenization, BertUnCasedPreTokenization}

Base.show(io::IO, ::BertCasedPreTokenization) = print(io, nameof(bert_cased_tokenizer))
Base.show(io::IO, ::BertUnCasedPreTokenization) = print(io, nameof(bert_uncased_tokenizer))

# encoder constructor

function BertTextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary{String}, process,
                         startsym::String, endsym::String, padsym::String, trunc::Union{Nothing, Int})
    return TransformerTextEncoder(tkr, vocab, process, startsym, endsym, padsym, trunc)
end

BertTextEncoder(::typeof(bert_cased_tokenizer), args...; kws...) =
    BertTextEncoder(BertCasedPreTokenization(), args...; kws...)
BertTextEncoder(::typeof(bert_uncased_tokenizer), args...; kws...) =
    BertTextEncoder(BertUnCasedPreTokenization(), args...; kws...)
BertTextEncoder(bt::BertTokenization, wordpiece::WordPiece, args...; kws...) =
    BertTextEncoder(WordPieceTokenization(bt, wordpiece), args...; kws...)
function BertTextEncoder(t::WordPieceTokenization, args...; match_tokens = nothing, kws...)
    if isnothing(match_tokens)
        return BertTextEncoder(TextTokenizer(t), Vocab(t.wordpiece), args...; kws...)
    else
        match_tokens = match_tokens isa AbstractVector ? match_tokens : [match_tokens]
        return BertTextEncoder(TextTokenizer(MatchTokenization(t, match_tokens)), Vocab(t.wordpiece), args...; kws...)
    end
end
function BertTextEncoder(t::AbstractTokenization, vocab::AbstractVocabulary, args...; match_tokens = nothing, kws...)
    if isnothing(match_tokens)
        return BertTextEncoder(TextTokenizer(t), vocab, args...; kws...)
    else
        match_tokens = match_tokens isa AbstractVector ? match_tokens : [match_tokens]
        return BertTextEncoder(TextTokenizer(MatchTokenization(t, match_tokens)), vocab, args...; kws...)
    end
end

function _wp_vocab(wp::WordPiece)
    vocab = Vector{String}(undef, length(wp.trie))
    for (str, id) in wp.trie
        vocab[wp.index[id]] = str
    end
    return vocab
end
TextEncodeBase.Vocab(wp::WordPiece) = Vocab(_wp_vocab(wp), DAT.decode(wp.trie, wp.unki))

function BertTextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary, process;
                         startsym = "[CLS]", endsym = "[SEP]", padsym = "[PAD]", trunc = nothing)
    check_vocab(vocab, startsym) || @warn "startsym $startsym not in vocabulary, this might cause problem."
    check_vocab(vocab, endsym) || @warn "endsym $endsym not in vocabulary, this might cause problem."
    check_vocab(vocab, padsym) || @warn "padsym $padsym not in vocabulary, this might cause problem."
    return BertTextEncoder(tkr, vocab, process, startsym, endsym, padsym, trunc)
end

function BertTextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary;
                         fixedsize = false, trunc_end = :tail, pad_end = :tail, process = nothing,
                         kws...)
    enc = BertTextEncoder(tkr, vocab, TextEncodeBase.process(AbstractTextEncoder); kws...)
    # default processing pipelines for bert encoder
    return BertTextEncoder(enc) do e
        bert_default_preprocess(; trunc = e.trunc, startsym = e.startsym, endsym = e.endsym, padsym = e.padsym,
                                fixedsize, trunc_end, pad_end, process)
    end
end

BertTextEncoder(builder, e::TrfTextEncoder) = TrfTextEncoder(builder, e)

# preprocess

function bert_default_preprocess(; startsym = "[CLS]", endsym = "[SEP]", padsym = "[PAD]",
                                 fixedsize = false, trunc = nothing, trunc_end = :tail, pad_end = :tail,
                                 process = nothing)
    truncf = get_trunc_pad_func(fixedsize, trunc, trunc_end, pad_end)
    maskf = get_mask_func(trunc, pad_end)
    if isnothing(process)
        process =
            # group input for SequenceTemplate
            Pipeline{:token}(grouping_sentence, :token) |>
            # add start & end symbol, compute segment and merge sentences
            Pipeline{:token_segment}(
                SequenceTemplate(
                    ConstTerm(startsym, 1), InputTerm{String}(1), ConstTerm(endsym, 1),
                    RepeatedTerm(InputTerm{String}(2), ConstTerm(endsym, 2); dynamic_type_id = true)
                ), :token
            ) |>
            Pipeline{:token}(nestedcall(first), :token_segment) |>
            Pipeline{:segment}(nestedcall(last), :token_segment)
    end
    # get token and convert to string
    return Pipeline{:token}(nestedcall(string_getvalue), 1) |>
        process |>
        Pipeline{:attention_mask}(maskf, :token) |>
        # truncate input that exceed length limit and pad them to have equal length
        Pipeline{:token}(truncf(padsym), :token) |>
        # convert to dense array
        Pipeline{:token}(nested2batch, :token) |>
        # truncate & pad segment
        Pipeline{:segment}(truncf(1), :segment) |>
        Pipeline{:segment}(nested2batch, :segment) |>
        # sequence mask
        Pipeline{:sequence_mask}(identity, :attention_mask) |>
        # return token and mask
        PipeGet{(:token, :segment, :attention_mask, :sequence_mask)}()
end
