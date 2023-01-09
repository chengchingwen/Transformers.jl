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
  ╰─ target[trunc_tok] := trunc_and_pad(5, [UNK])(target.tok)
  ╰─ target[trunc_len] := nestedmaxlength(target.trunc_tok)
  ╰─ target[mask] := getmask(target.tok, target.trunc_len)
  ╰─ target[tok] := nested2batch(target.trunc_tok)
  ╰─ target[segment] := (nested2batch ∘ trunc_and_pad(5, 1))(target.segment)
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
├─ trunc = 5,
└─ process = Pipelines:
  ╰─ target[tok] := nestedcall(string_getvalue, source)
  ╰─ target[tok] := with_firsthead_tail([CLS], [SEP])(target.tok)
  ╰─ target[(tok, segment)] := segment_and_concat(target.tok)
  ╰─ target := (target.tok, target.segment)
)

```
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
 `@NamedTuple{input::@NamedTuple{tok::OneHotArray{K, 2}, segment::Vector{Int}}, mask::Nothing}`

    encode(::BertTextEncoder, ::Vector{String})

Encode a batch of sentences with bert text encoder. The default pipeline returning
 `@NamedTuple{input::@NamedTuple{tok::OneHotArray{K, 3}, segment::Matrix{Int}}, mask::Array{Float32, 3}}`

    encode(::BertTextEncoder, ::Vector{Vector{String}})

Encode a batch of segments with bert text encoder. The default pipeline returning
 `@NamedTuple{input::@NamedTuple{tok::OneHotArray{K, 3}, segment::Matrix{Int}}, mask::Array{Float32, 3}}`

See also: [`decode`](@ref)

# Example
```julia-repl
julia> wordpiece = pretrain"bert-cased_L-12_H-768_A-12:wordpiece";
[ Info: loading pretrain bert model: cased_L-12_H-768_A-12.tfbson wordpiece

julia> bertenc = BertTextEncoder(bert_cased_tokenizer, wordpiece);

julia> e = encode(bertenc, [["this is a sentence", "and another"]])
(input = (tok = [0 0 … 0 0; 0 0 … 0 0; … ; 0 0 … 0 0; 0 0 … 0 0;;;], segment = [1; 1; … ; 2; 2;;]), mask = [1.0 1.0 … 1.0 1.0;;;])

julia> typeof(e)
NamedTuple{(:input, :mask), Tuple{NamedTuple{(:token, :segment), Tuple{OneHotArray{0x00007144, 2, 3, Matrix{OneHot{0x00007144}}}, Matrix{Int64}}}, Array{Float32, 3}}}

```
"""
TextEncodeBase.encode(::BertTextEncoder, _)

"""
    decode(bertenc::BertTextEncoder, x)

Equivalent to `lookup(bertenc.vocab, x)`.

See also: [`encode`](@ref)

# Example
```julia-repl
julia> tok = encode(bertenc, [["this is a sentence", "and another"]]).input.tok;

julia> decode(bertenc, tok)
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

julia> lookup(bertenc.vocab, tok)
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

# pretty print

Base.show(io::IO, ::BertCasedPreTokenization) = print(io, nameof(bert_cased_tokenizer))
Base.show(io::IO, ::BertUnCasedPreTokenization) = print(io, nameof(bert_uncased_tokenizer))
Base.show(io::IO, wp::WordPieceTokenization) = print(io, "WordPieceTokenization(", wp.base, ", ", wp.wordpiece, ')')
