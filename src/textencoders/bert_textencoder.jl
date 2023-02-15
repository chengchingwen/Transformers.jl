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

# encoder

"""
    BertTextEncoder

The text encoder for Bert model (WordPiece tokenization).
"""
struct BertTextEncoder{T <: AbstractTokenizer, V <: AbstractVocabulary{String}, P} <: AbstractTransformerTextEncoder
    tokenizer::T
    vocab::V
    process::P
    startsym::String
    endsym::String
    padsym::String
    trunc::Union{Nothing, Int}
end

# encoder constructor

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

BertTextEncoder(builder, e::BertTextEncoder) =
    BertTextEncoder(e.tokenizer, e.vocab, builder(e), e.startsym, e.endsym, e.padsym, e.trunc)

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
        # return input and mask
        PipeGet{(:token, :segment, :attention_mask)}()
end

# api doc

"""
    encode(::BertTextEncoder, ::String)

Encode a single sentence with bert text encoder. The default pipeline returning
 `@NamedTuple{token::OneHotArray{K, 1}, segment::Vector{Int}, attention_mask::LengthMask{1, Vector{Int32}}}`.

    encode(::BertTextEncoder, ::Vector{String})

Encode a batch of sentences with bert text encoder. The default pipeline returning
 `@NamedTuple{token::OneHotArray{K, 2}, segment::Matrix{Int}, attention_mask::LengthMask{1, Vector{Int32}}}`.

    encode(::BertTextEncoder, ::Vector{Vector{String}})

Encode a batch of segments with bert text encoder. Segments would be concatenate together as batch of sentences with
 separation token and correct indicator in `segment`. The default pipeline returning
 `@NamedTuple{token::OneHotArray{K, 2}, segment::Matrix{Int}, attention_mask::LengthMask{1, Vector{Int32}}}`.

    encode(::BertTextEncoder, ::Vector{Vector{Vector{String}}})

Encode a batch of multi-sample segments with bert text encoder. The number of sample per data need to be the same.
 (e.g. `length(batch[1]) == length(batch[2])`). The default pipeline returning
 `@NamedTuple{token::OneHotArray{K, 3}, segment::Array{Int, 3}, attention_mask::LengthMask{2, Matrix{Int32}}}`.
 *notice*: If you want each sample to be independent to each other, this need to be reshaped before feeding to
 transformer layer or make sure the attention is not taking the `end-1` dimension as another length dimension.

See also: [`decode`](@ref), `LengthMask`

# Example
```julia-repl
julia> bertenc = HuggingFace.load_tokenizer("bert-base-cased")
BertTextEncoder(
├─ TextTokenizer(MatchTokenization(WordPieceTokenization(bert_cased_tokenizer, WordPiece(vocab_size = 28996, unk = [UNK], max_char = 100)), 5 patterns)),
├─ vocab = Vocab{String, SizedArray}(size = 28996, unk = [UNK], unki = 101),
├─ startsym = [CLS],
├─ endsym = [SEP],
├─ padsym = [PAD],
├─ trunc = 512,
└─ process = Pipelines:
  ╰─ target[token] := TextEncodeBase.nestedcall(string_getvalue, source)
  ╰─ target[token] := Transformers.TextEncoders.grouping_sentence(target.token)
  ╰─ target[(token, segment)] := SequenceTemplate{String}([CLS]:<type=1> Input[1]:<type=1> [SEP]:<type=1> (Input[2]:<type=2> [SEP]:<type=2>)...)(target.token)
  ╰─ target[attention_mask] := (NeuralAttentionlib.LengthMask ∘ Transformers.TextEncoders.getlengths(512))(target.token)
  ╰─ target[token] := TextEncodeBase.trunc_and_pad(512, [PAD], tail, tail)(target.token)
  ╰─ target[token] := TextEncodeBase.nested2batch(target.token)
  ╰─ target[segment] := TextEncodeBase.trunc_and_pad(512, 1, tail, tail)(target.segment)
  ╰─ target[segment] := TextEncodeBase.nested2batch(target.segment)
  ╰─ target := (target.token, target.segment, target.attention_mask)
)

julia> e = encode(bertenc, [["this is a sentence", "and another"]])
(token = [0 0 … 0 0; 0 0 … 0 0; … ; 0 0 … 0 0; 0 0 … 0 0;;;], segment = [1; 1; … ; 2; 2;;], attention_mask = NeuralAttentionlib.LengthMask{1, Vector{Int32}}(Int32[9]))

julia> typeof(e)
NamedTuple{(:token, :segment, :attention_mask), Tuple{OneHotArray{0x00007144, 2, 3, Matrix{OneHot{0x00007144}}}, Matrix{Int64}, NeuralAttentionlib.LengthMask{1, Vector{Int32}}}}

```
"""
TextEncodeBase.encode(::BertTextEncoder, _)

"""
    decode(bertenc::BertTextEncoder, x)

Convert indices back to string with bert vocabulary.

See also: [`encode`](@ref)

# Example
```julia-repl
julia> token = encode(bertenc, [["this is a sentence", "and another"]]).token;

julia> decode(bertenc, token)
9×1 Matrix{String}:
 "[CLS]"
 "this"
 "is"
 "a"
 "sentence"
 "[SEP]"
 "and"
 "another"
 "[SEP]"

```
"""
TextEncodeBase.decode(::BertTextEncoder, _)
