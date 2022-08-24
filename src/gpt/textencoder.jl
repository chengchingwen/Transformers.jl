using ..Basic: string_getvalue, check_vocab, TextTokenizer, WList, concat, with_firsthead_tail
using FuncPipelines
using TextEncodeBase
using TextEncodeBase: trunc_and_pad, trunc_or_pad, nested2batch, nestedcall
using TextEncodeBase: BaseTokenization, WrappedTokenization, MatchTokenization, CodeNormalizer,
    ParentStages, TokenStages, SentenceStage, WordStage, Batch, Sentence, getvalue, getmeta

using BytePairEncoding
using BytePairEncoding: AbstractBPE

# gpt tokenizer

struct GPTTokenization <: BaseTokenization end

TextEncodeBase.splitting(::GPTTokenization, s::SentenceStage) = gpt_tokenizer(getvalue(s))

using BytePairEncoding
using BytePairEncoding: GPT2Tokenization

# encoder

struct GPTTextEncoder{T<:AbstractTokenizer, V<:AbstractVocabulary{String}, P} <: AbstractTextEncoder
    tokenizer::T
    vocab::V
    process::P
    startsym::String
    sepsym::String
    endsym::String
    padsym::String
    trunc::Union{Nothing, Int}
end

struct GPT2TextEncoder{T<:AbstractTokenizer, V<:AbstractVocabulary{String}, P} <: AbstractTextEncoder
    tokenizer::T
    vocab::V
    process::P
    startsym::String
    sepsym::String
    endsym::String
    padsym::String
    trunc::Union{Nothing, Int}
end

# encoder constructor

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

function GPTTextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary;
                        fixedsize = false, trunc_end = :head, pad_end = :head,
                        kwargs...)
    enc = GPTTextEncoder(tkr, vocab, TextEncodeBase.process(AbstractTextEncoder); kwargs...)
    # default processing pipelines for bert encoder
    return GPTTextEncoder(enc) do e
        gpt_default_preprocess(; trunc = e.trunc, startsym = e.startsym, endsym = e.endsym, padsym = e.padsym,
                               fixedsize, trunc_end, pad_end)
    end
end

GPTTextEncoder(builder, e::GPTTextEncoder) =
    GPTTextEncoder(e.tokenizer, e.vocab, builder(e), e.startsym, e.sepsym, e.endsym, e.padsym, e.trunc)

# preprocess

function gpt_default_preprocess(; startsym = "_start_", sepsym = "_delimiter_", endsym = "_classify_",
                                unksym = "<unk>", padsym = "<pad>", trunc = nothing, fixedsize = false,
                                trunc_end = :head, pad_end = :head)
    if fixedsize
        @assert !isnothing(trunc) "`fixedsize=true` but `trunc` is not set."
        truncf = trunc_or_pad
    else
        truncf = trunc_and_pad
    end

    return Pipeline{:tok}(nestedcall(string_getvalue), 1) |>
        # add start & end symbol
        Pipeline{:tok}(with_firsthead_tail(startsym, endsym, sepsym), :tok) |>
        # compute segment and merge sentences
        Pipeline{:tok}(concat, :tok) |>
        # truncate input that exceed length limit and pad them to have equal length
        Pipeline{:trunc_tok}(truncf(trunc, padsym, trunc_end, pad_end), :tok) |>
        # get the truncated length
        (fixedsize ?
         Pipeline{:trunc_len}(FuncPipelines.FixRest(identity, trunc), 0) :
         Pipeline{:trunc_len}(TextEncodeBase.nestedmaxlength, :trunc_tok)
         ) |>
        # set pad end
        Pipeline{:lpad}(FuncPipelines.FixRest(identity, pad_end == :head), 0) |>
        # get mask with specific length
        Pipeline{:mask}(getmask, (:tok, :trunc_len, :lpad)) |>
        # convert to dense array
        Pipeline{:tok}(nested2batch, :trunc_tok) |>
        # input namedtuple
        Pipeline{:input}(NamedTuple{(:tok,)}∘tuple, :tok) |>
        # return input and mask
        PipeGet{(:input, :mask)}()
end

# encoder behavior

TextEncodeBase.tokenize(e::GPTTextEncoder, x::AbstractString) = e.tokenizer(Sentence(x))
TextEncodeBase.tokenize(e::GPTTextEncoder, x::Vector{<:AbstractString}) = e.tokenizer(Batch{Sentence}(x))
TextEncodeBase.tokenize(e::GPTTextEncoder, x::Vector{<:Vector{<:AbstractString}}) = e.tokenizer(Batch{Batch{Sentence}}(x))

function TextEncodeBase.lookup(e::GPTTextEncoder, x::NamedTuple)
    onehot_tok = lookup(e, x.input.tok)
    input = merge(x.input, (tok = onehot_tok,))
    return merge(x, (input = input,))
end

# pretty print

function Base.show(io::IO, e::GPTTextEncoder)
    print(io, "GPTTextEncoder(\n├─ ")
    print(io, e.tokenizer, ",\n├─ ")
    print(io, "vocab = ", e.vocab, ",\n├─ ")
    print(io, "startsym = ", e.startsym, ",\n├─ ")
    print(io, "sepsym = ", e.sepsym, ",\n├─ ")
    print(io, "endsym = ", e.endsym, ",\n├─ ")
    print(io, "padsym = ", e.padsym)
    isnothing(e.trunc) || print(io, ",\n├─ trunc = ", e.trunc)
    print(IOContext(io, :pipeline_display_prefix => "  ╰─ "), ",\n└─ process = ", e.process, "\n)")
end

Base.show(io::IO, ::GPTTokenization) = print(io, nameof(gpt_tokenizer))
