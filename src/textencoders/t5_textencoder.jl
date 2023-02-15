using TextEncodeBase
using TextEncodeBase: trunc_and_pad, trunc_or_pad, nested2batch, nestedcall, Batch, Sentence
using TextEncodeBase: SequenceTemplate, ConstTerm, InputTerm, RepeatedTerm
using FuncPipelines


"""
    T5TextEncoder

The text encoder for T5 model (SentencePiece tokenization).
"""
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

# api doc

"""
    encode(::T5TextEncoder, ::String)

Encode a single sentence with t5 text encoder. The default pipeline returning
 `@NamedTuple{token::OneHotArray{K, 1}, attention_mask::LengthMask{1, Vector{Int32}}}`.

    encode(::T5TextEncoder, ::Vector{String})

Encode a batch of sentences with t5 text encoder. The default pipeline returning
 `@NamedTuple{token::OneHotArray{K, 2}, attention_mask::LengthMask{1, Vector{Int32}}}`.

    encode(::T5TextEncoder, ::Vector{Vector{String}})

Encode a batch of segments with t5 text encoder. Segments would be concatenate together as batch of sentences with a
 separation token. The default pipeline returning
 `@NamedTuple{token::OneHotArray{K, 2}, attention_mask::LengthMask{1, Vector{Int32}}}`.

    encode(::T5TextEncoder, ::Vector{Vector{Vector{String}}})

Encode a batch of multi-sample segments with t5 text encoder. The number of sample per data need to be the same.
 (e.g. `length(batch[1]) == length(batch[2])`). The default pipeline returning
 `@NamedTuple{token::OneHotArray{K, 3}, attention_mask::LengthMask{2, Matrix{Int32}}}`.
 *notice*: If you want each sample to be independent to each other, this need to be reshaped before feeding to
 transformer layer or make sure the attention is not taking the `end-1` dimension as another length dimension.

See also: [`decode`](@ref), `LengthMask`

# Example
```julia-repl
julia> t5enc = HuggingFace.load_tokenizer("t5")
T5TextEncoder(
├─ TextTokenizer(MatchTokenization(PrecompiledNormalizer(WordReplaceNormalizer(UnigramTokenization(EachSplitTokenization(splitter = isspace), unigram = Unigram(vocab_size = 32100, unk = <unk>)), pattern = r"^(?!▁)(.*)\$" => s"▁\1"), precompiled
= PrecompiledNorm(...)), 103 patterns)),
├─ vocab = Vocab{String, SizedArray}(size = 32100, unk = <unk>, unki = 3),
├─ endsym = </s>,
├─ padsym = <pad>,
└─ process = Pipelines:
  ╰─ target[token] := TextEncodeBase.nestedcall(string_getvalue, source)
  ╰─ target[token] := Transformers.TextEncoders.grouping_sentence(target.token)
  ╰─ target[(token, segment)] := SequenceTemplate{String}(Input[1]:<type=1> </s>:<type=1> (Input[2]:<type=1> </s>:<type=1>)...)(target.token)
  ╰─ target[attention_mask] := (NeuralAttentionlib.LengthMask ∘ Transformers.TextEncoders.getlengths(nothing))(target.token)
  ╰─ target[token] := TextEncodeBase.trunc_and_pad(nothing, <pad>, tail, tail)(target.token)
  ╰─ target[token] := TextEncodeBase.nested2batch(target.token)
  ╰─ target := (target.token, target.attention_mask)
)

julia> e = encode(t5enc, [["this is a sentence", "and another"]])
(token = [0 0 … 0 0; 0 0 … 0 1; … ; 0 0 … 0 0; 0 0 … 0 0;;;], attention_mask = NeuralAttentionlib.LengthMask{1, Vector{Int32}}(Int32[9]))

julia> typeof(e)
NamedTuple{(:token, :attention_mask), Tuple{OneHotArray{0x00007d64, 2, 3, Matrix{OneHot{0x00007d64}}}, NeuralAttentionlib.LengthMask{1, Vector{Int32}}}}

```
"""
TextEncodeBase.encode(::T5TextEncoder, _)

"""
    decode(bertenc::T5TextEncoder, x)

Convert indices back to string with t5 vocabulary.

See also: [`encode`](@ref)

# Example
```julia-repl
julia> token = encode(t5enc, [["this is a sentence", "and another"]]).token;

julia> decode(t5enc, token)
9×1 Matrix{String}:
 "▁this"
 "▁is"
 "▁"
 "a"
 "▁sentence"
 "</s>"
 "▁and"
 "▁another"
 "</s>"

```
"""
TextEncodeBase.decode(::T5TextEncoder, _)
