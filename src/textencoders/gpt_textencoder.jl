using StructWalk
using FuncPipelines
using TextEncodeBase
using TextEncodeBase: trunc_and_pad, trunc_or_pad, nested2batch, nestedcall
using TextEncodeBase: BaseTokenization, WrappedTokenization, MatchTokenization, CodeNormalizer,
    CodeMap, CodeUnMap, ParentStages, TokenStages, SentenceStage, WordStage, Batch, Sentence, getvalue, getmeta
using TextEncodeBase: SequenceTemplate, ConstTerm, InputTerm, RepeatedTerm

using BytePairEncoding
using BytePairEncoding: AbstractBPE, gpt2_codemap

function find_codemap(tkr)
    rcm = Ref{Union{Nothing, CodeMap}}(nothing)
    StructWalk.scan(x->x isa CodeMap && (rcm[] = x), TextEncodeBase.TokenizerStyle(), tkr)
    cm = rcm[]
    isnothing(cm) && error("cannot find codemap from gpt2 text encoder.")
    return cm
end

# gpt tokenizer

struct GPTTokenization <: BaseTokenization end

TextEncodeBase.splitting(::GPTTokenization, s::SentenceStage) = gpt_tokenizer(getvalue(s))

Base.show(io::IO, ::GPTTokenization) = print(io, nameof(gpt_tokenizer))

## bpe tokenization and gpt2 tokenizer

using BytePairEncoding
using BytePairEncoding: GPT2Tokenization, gpt2_tokenizer

# encoder constructor

function GPTTextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary{String}, process,
                        startsym::Union{Nothing, String}, sepsym::Union{Nothing, String}, endsym::Union{Nothing, String},
                        padsym::Union{Nothing, String}, trunc::Union{Nothing, Int})
    return TrfTextEncoder(
        tkr, vocab,
        @NamedTuple{startsym::Union{Nothing, String}, sepsym::Union{Nothing, String}, endsym::Union{Nothing, String},
                    padsym::Union{Nothing, String}, trunc::Union{Nothing, Int}}(
                        (startsym, sepsym, endsym, padsym, trunc)),
        annotate_strings,
        process,
        lookup_first,
        identity,
        TextEncodeBase.join_text,
    )
end

GPTTextEncoder(::typeof(gpt_tokenizer), args...; kwargs...) =
    GPTTextEncoder(GPTTokenization(), args...; kwargs...)
GPTTextEncoder(gt::GPTTokenization, bpe::AbstractBPE, args...; kwargs...) =
    GPTTextEncoder(BPETokenization(gt, bpe), args...; kwargs...)
function GPTTextEncoder(t::AbstractTokenization, vocab::WList, args...; match_tokens = nothing, kwargs...)
    if isnothing(match_tokens)
        return GPTTextEncoder(TextTokenizer(t), vocab, args...; kwargs...)
    else
        match_tokens = match_tokens isa AbstractVector ? match_tokens : [match_tokens]
        return GPTTextEncoder(TextTokenizer(MatchTokenization(t, match_tokens)), vocab, args...; kwargs...)
    end
end

function GPTTextEncoder(tkr::AbstractTokenizer, words::AbstractVector, process;
                        startsym = "_start_", sepsym = "_delimiter_", endsym = "_classify_",
                        unksym = "<unk>", padsym = "<pad>", trunc = nothing)
    vocab_list = copy(words)
    for sym in (padsym, unksym, startsym, endsym)
        sym ∉ vocab_list && push!(vocab_list, sym)
    end
    vocab = Vocab(vocab_list, unksym)
    return GPTTextEncoder(tkr, vocab, process, startsym, sepsym, endsym, padsym, trunc)
end

function GPTTextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary, process;
                        startsym = "_start_", sepsym = "_delimiter_", endsym = "_classify_",
                        unksym = "<unk>", padsym = "<pad>", trunc = nothing)
    check_vocab(vocab, startsym) || @warn "startsym $startsym not in vocabulary, this might cause problem."
    check_vocab(vocab, sepsym) || @warn "sepsym $sepsym not in vocabulary, this might cause problem."
    check_vocab(vocab, endsym) || @warn "endsym $endsym not in vocabulary, this might cause problem."
    check_vocab(vocab, unksym) || @warn "unksym $unksym not in vocabulary, this might cause problem."
    check_vocab(vocab, padsym) || @warn "padsym $padsym not in vocabulary, this might cause problem."
    return GPTTextEncoder(tkr, vocab, process, startsym, sepsym, endsym, padsym, trunc)
end

function GPTTextEncoder(tkr::AbstractTokenizer, vocab::WList;
                        fixedsize = false, trunc_end = :head, pad_end = :head, process = nothing,
                        kwargs...)
    enc = GPTTextEncoder(tkr, vocab, TextEncodeBase.process(AbstractTextEncoder); kwargs...)
    # default processing pipelines for bert encoder
    return GPTTextEncoder(enc) do e
        gpt_default_preprocess(; trunc = e.trunc, startsym = e.startsym, sepsym = e.sepsym,
                               endsym = e.endsym, padsym = e.padsym,
                               fixedsize, trunc_end, pad_end, process)
    end
end

GPTTextEncoder(builder, e::TrfTextEncoder) = TrfTextEncoder(builder, e)

## gpt2 encoder constructor

function GPT2TextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary{String}, process,
                         codemap::CodeMap, startsym::Union{Nothing, String}, endsym::Union{Nothing, String},
                         padsym::Union{Nothing, String}, trunc::Union{Nothing, Int})
    return TrfTextEncoder(
        tkr, vocab,
        @NamedTuple{
            codemap::typeof(codemap), startsym::Union{Nothing, String}, endsym::Union{Nothing, String},
            padsym::Union{Nothing, String}, trunc::Union{Nothing, Int}}(
                (codemap, startsym, endsym, padsym, trunc)),
        annotate_strings,
        process,
        lookup_first,
        TextEncodeBase.nestedcall(CodeUnMap(codemap)),
        TextEncodeBase.join_text,
    )
end

GPT2TextEncoder(::typeof(gpt2_tokenizer), args...; kwargs...) =
    GPT2TextEncoder(GPT2Tokenization(), args...; kwargs...)
GPT2TextEncoder(gt::GPT2Tokenization, bpe::AbstractBPE, args...; kwargs...) =
    GPT2TextEncoder(BPETokenization(gt, bpe), args...; kwargs...)
GPT2TextEncoder(bt::BPETokenization, cm::CodeMap, args...; kwargs...) =
    GPT2TextEncoder(CodeNormalizer(bt, cm), args...; kwargs...)
GPT2TextEncoder(bt::BPETokenization, vocab::WList, args...; kwargs...) =
    GPT2TextEncoder(CodeNormalizer(bt, gpt2_codemap()), args...; kwargs...)
function GPT2TextEncoder(t::AbstractTokenization, vocab::WList, args...; match_tokens = ["<|endoftext|>"], kwargs...)
    if isnothing(match_tokens)
        return GPT2TextEncoder(TextTokenizer(t), vocab, args...; kwargs...)
    else
        match_tokens = match_tokens isa AbstractVector ? match_tokens : [match_tokens]
        return GPT2TextEncoder(TextTokenizer(MatchTokenization(t, match_tokens)), vocab, args...; kwargs...)
    end
end

function GPT2TextEncoder(tkr::AbstractTokenizer, words::AbstractVector, process;
                         startsym = "<|endoftext|>", endsym = "<|endoftext|>",
                         unksym = "<|endoftext|>", padsym = "<|endoftext|>", trunc = nothing)
    vocab_list = copy(words)
    for sym in (padsym, unksym, startsym, endsym)
        sym ∉ vocab_list && push!(vocab_list, sym)
    end
    vocab = Vocab(vocab_list, unksym)
    return GPT2TextEncoder(tkr, vocab, process, find_codemap(tkr), startsym, endsym, padsym, trunc)
end

function GPT2TextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary, process;
                         startsym = "<|endoftext|>", endsym = "<|endoftext|>",
                         unksym = "<|endoftext|>", padsym = "<|endoftext|>", trunc = nothing)
    check_vocab(vocab, startsym) || @warn "startsym $startsym not in vocabulary, this might cause problem."
    check_vocab(vocab, endsym) || @warn "endsym $endsym not in vocabulary, this might cause problem."
    check_vocab(vocab, unksym) || @warn "unksym $unksym not in vocabulary, this might cause problem."
    check_vocab(vocab, padsym) || @warn "padsym $padsym not in vocabulary, this might cause problem."
    return GPT2TextEncoder(tkr, vocab, process, find_codemap(tkr), startsym, endsym, padsym, trunc)
end

function GPT2TextEncoder(tkr::AbstractTokenizer, vocab::WList;
                        fixedsize = false, trunc_end = :head, pad_end = :head, process = nothing,
                        kwargs...)
    enc = GPT2TextEncoder(tkr, vocab, TextEncodeBase.process(AbstractTextEncoder); kwargs...)
    # default processing pipelines for bert encoder
    return GPT2TextEncoder(enc) do e
        gpt2_default_preprocess(; trunc = e.trunc, startsym = e.startsym, endsym = e.endsym, padsym = e.padsym,
                                fixedsize, trunc_end, pad_end, process)
    end
end

GPT2TextEncoder(builder, e::TrfTextEncoder) = TrfTextEncoder(builder, e)

# preprocess

function gpt_default_preprocess(; startsym = "_start_", sepsym = "_delimiter_", endsym = "_classify_",
                                padsym = "<pad>", trunc = nothing, fixedsize = false,
                                trunc_end = :head, pad_end = :head, process = nothing)
    truncf = get_trunc_pad_func(padsym, fixedsize, trunc, trunc_end, pad_end)
    maskf = get_mask_func(trunc, pad_end)
    if isnothing(process)
        process =
            # group input for SequenceTemplate
            Pipeline{:token}(grouping_sentence, :token) |>
            # add start & end symbol and merge sentences
            Pipeline{:token}(SequenceTemplate(
                ConstTerm(startsym), InputTerm{String}(),
                RepeatedTerm(ConstTerm(sepsym), InputTerm{String}()),
                ConstTerm(endsym),
            )(Val(1)), :token)
    end
    return Pipeline{:token}(nestedcall(string_getvalue), 1) |>
        process |>
        # get mask with specific length
        Pipeline{:attention_mask}(maskf, :token) |>
        # truncate input that exceed length limit and pad them to have equal length
        Pipeline{:token}(truncf, :token) |>
        # convert to dense array
        Pipeline{:token}(nested2batch, :token) |>
        # sequence mask
        Pipeline{:sequence_mask}(identity, :attention_mask) |>
        # return token and mask
        PipeGet{(:token, :attention_mask, :sequence_mask)}()
end

function gpt2_default_preprocess(; startsym = "<|endoftext|>", endsym = "<|endoftext|>", padsym = "<|endoftext|>",
                                 trunc = nothing, fixedsize = false, trunc_end = :head, pad_end = :head,
                                 process = nothing)
    truncf = get_trunc_pad_func(padsym, fixedsize, trunc, trunc_end, pad_end)
    maskf = get_mask_func(trunc, pad_end)
    if isnothing(process)
        process =
            # group input for SequenceTemplate
            Pipeline{:token}(grouping_sentence, :token) |>
            # add start & end symbol and merge sentences
            Pipeline{:token}(SequenceTemplate(RepeatedTerm(InputTerm{String}()))(Val(1)), :token)
    end

    return Pipeline{:token}(nestedcall(string_getvalue), 1) |>
        process |>
        # get mask with specific length
        Pipeline{:attention_mask}(maskf, :token) |>
        # truncate input that exceed length limit and pad them to have equal length
        Pipeline{:token}(truncf, :token) |>
        # convert to dense array
        Pipeline{:token}(nested2batch, :token) |>
        # sequence mask
        Pipeline{:sequence_mask}(identity, :attention_mask) |>
        # return token and mask
        PipeGet{(:token, :attention_mask, :sequence_mask)}()
end
