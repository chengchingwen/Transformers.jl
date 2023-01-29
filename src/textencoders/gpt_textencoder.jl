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

# encoder

struct GPTTextEncoder{T <: AbstractTokenizer, V <: AbstractVocabulary{String}, P} <: AbstractTransformerTextEncoder
    tokenizer::T
    vocab::V
    process::P
    startsym::Union{Nothing, String}
    sepsym::Union{Nothing, String}
    endsym::Union{Nothing, String}
    padsym::Union{Nothing, String}
    trunc::Union{Nothing, Int}
end

## gpt2 encoder

"""
    GPT2TextEncoder

The text encoder for GPT2 model (ByteLevel BytePairEncoding tokenization).
"""
struct GPT2TextEncoder{T <: AbstractTokenizer, V <: AbstractVocabulary{String}, P, C<:CodeMap} <: AbstractTransformerTextEncoder
    tokenizer::T
    vocab::V
    process::P
    codemap::C
    startsym::Union{Nothing, String}
    endsym::Union{Nothing, String}
    padsym::Union{Nothing, String}
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

GPTTextEncoder(builder, e::GPTTextEncoder) =
    GPTTextEncoder(e.tokenizer, e.vocab, builder(e), e.startsym, e.sepsym, e.endsym, e.padsym, e.trunc)

## gpt2 encoder constructor

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

GPT2TextEncoder(builder, e::GPT2TextEncoder) =
    GPT2TextEncoder(e.tokenizer, e.vocab, builder(e), e.codemap, e.startsym, e.endsym, e.padsym, e.trunc)


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
        # return input and mask
        PipeGet{(:token, :attention_mask)}()
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
        # return input and mask
        PipeGet{(:token, :attention_mask)}()
end

# decode

function TextEncodeBase.decode(e::GPT2TextEncoder, i::Union{Integer, OneHotArray, AbstractArray{<:Integer}})
    uc = CodeUnMap(e.codemap)
    return TextEncodeBase.nestedcall(uc, TextEncodeBase.decode_indices(e, i))
end

# api doc

"""
    encode(::GPT2TextEncoder, ::String)

Encode a single sentence with gpt2 text encoder. The default pipeline returning
 `@NamedTuple{token::OneHotArray{K, 1}, attention_mask::RevLengthMask{1, Vector{Int32}}}`.

    encode(::GPT2TextEncoder, ::Vector{String})

Encode a batch of sentences with gpt2 text encoder. The default pipeline returning
 `@NamedTuple{token::OneHotArray{K, 2}, attention_mask::RevLengthMask{1, Vector{Int32}}}`.

    encode(::GPT2TextEncoder, ::Vector{Vector{String}})

Encode a batch of segments with gpt2 text encoder. Segments would be concatenate together as batch of sentences.
 The default pipeline returning `@NamedTuple{token::OneHotArray{K, 2}, attention_mask::RevLengthMask{1, Vector{Int32}}}`.

    encode(::GPT2TextEncoder, ::Vector{Vector{Vector{String}}})

Encode a batch of multi-sample segments with gpt2 text encoder. The number of sample per data need to be the same.
 (e.g. `length(batch[1]) == length(batch[2])`). The default pipeline returning
 `@NamedTuple{token::OneHotArray{K, 3}, attention_mask::RevLengthMask{2, Matrix{Int32}}}`.
 *notice*: If you want each sample to be independent to each other, this need to be reshaped before feeding to
 transformer layer or make sure the attention is not taking the `end-1` dimension as another length dimension.

See also: [`decode`](@ref), `RevLengthMask`

# Example
```julia-repl
julia> gpt2enc = HuggingFace.load_tokenizer("gpt2")
GPT2TextEncoder(
├─ TextTokenizer(MatchTokenization(CodeNormalizer(BPETokenization(GPT2Tokenization, bpe = CachedBPE(BPE(50000 merges))), codemap = CodeMap{UInt8 => UInt16}(3 code-ranges)), 1 patterns)),
├─ vocab = Vocab{String, SizedArray}(size = 50257, unk = <unk>, unki = 0),
├─ codemap = CodeMap{UInt8 => UInt16}(3 code-ranges),
├─ startsym = <|endoftext|>,
├─ endsym = <|endoftext|>,
├─ padsym = <|endoftext|>,
├─ trunc = 1024,
└─ process = Pipelines:
  ╰─ target[token] := TextEncodeBase.nestedcall(string_getvalue, source)
  ╰─ target[token] := Transformers.TextEncoders.grouping_sentence(target.token)
  ╰─ target[token] := SequenceTemplate{String}((Input:<type=1>)...)(Val{1}(), target.token)
  ╰─ target[attention_mask] := (NeuralAttentionlib.RevLengthMask ∘ Transformers.TextEncoders.getlengths(1024))(target.token)
  ╰─ target[token] := TextEncodeBase.trunc_and_pad(1024, <|endoftext|>, head, head)(target.token)
  ╰─ target[token] := TextEncodeBase.nested2batch(target.token)
  ╰─ target := (target.token, target.attention_mask)
)

julia> e = encode(gpt2enc, [["this is a sentence", "and another"]])
(token = [0 0 … 0 0; 0 0 … 0 0; … ; 0 0 … 0 0; 0 0 … 0 0;;;], attention_mask = NeuralAttentionlib.RevLengthMask{1, Vector{Int32}}(Int32[6]))

julia> typeof(e)
NamedTuple{(:token, :attention_mask), Tuple{OneHotArray{0x0000c451, 2, 3, Matrix{OneHot{0x0000c451}}}, NeuralAttentionlib.RevLengthMask{1, Vector{Int32}}}}

```
"""
TextEncodeBase.encode(::GPT2TextEncoder, _)

"""
    decode(bertenc::GPT2TextEncoder, x)

Convert indices back to string with gpt2 vocabulary. This would also map the bytes back to the normal code ranges,
 so the string is not directly the one in the vocabulary.

See also: [`encode`](@ref)

# Example
```julia-repl
julia> token = encode(gpt2enc, [["this is a sentence", "and another"]]).token;

julia> decode(gpt2enc, token)
6×1 Matrix{String}:
 "this"
 " is"
 " a"
 " sentence"
 "and"
 " another"

julia> TextEncodeBase.decode_indices(gpt2enc, token)
6×1 Matrix{String}:
 "this"
 "Ġis"
 "Ġa"
 "Ġsentence"
 "and"
 "Ġanother"

```
"""
TextEncodeBase.decode(::GPT2TextEncoder, _)
