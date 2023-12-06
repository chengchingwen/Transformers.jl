module TextEncoders

using Tricks
using PrimitiveOneHot
using FuncPipelines
using TextEncodeBase
using TextEncodeBase: WordTokenization, nested2batch, nestedcall, with_head_tail, tokenize, decode_text
using ..WordPieceModel
using BytePairEncoding
using ..UnigramLanguageModel

using NeuralAttentionlib: AttenMask, LengthMask, RevLengthMask, GenericSequenceMask

export lookup, encode, decode, decode_text, Vocab, OneHot, OneHotArray,
    TransformerTextEncoder, BertTextEncoder, GPT2TextEncoder, T5TextEncoder


include("bert_tokenizer.jl")
include("gpt_tokenizer.jl")

include("utils.jl")
include("tokenizer.jl")

abstract type AbstractTransformerTextEncoder <: AbstractTextEncoder end

function Base.show(io::IO, e::AbstractTransformerTextEncoder)
    print(io, "$(nameof(typeof(e)))(\n├─ ")
    print(io, e.tokenizer)
    _io = IOContext(io, :compact => true)
    for name in fieldnames(typeof(e))
        (name == :tokenizer || name == :process) && continue
        val = getfield(e, name)
        if !isnothing(val)
            if val isa Base.Fix1{typeof(nestedcall)}
                print(_io, ",\n├─ ", name, " = nestedcall")
                val.x isa ComposedFunction || print(_io, '(')
                FuncPipelines.show_pipeline_function(_io, val.x)
                val.x isa ComposedFunction || print(_io, ')')
            elseif !(val isa Pipelines) && val isa Function
                print(_io, ",\n├─ ", name, " = ")
                FuncPipelines.show_pipeline_function(_io, val)
            else
                print(_io, ",\n├─ ", name, " = ", val)
            end
        end
    end
    print(IOContext(io, :pipeline_display_prefix => "  ╰─ "), ",\n└─ process = ", e.process, "\n)")
end

struct TrfTextEncoder{
    T <: AbstractTokenizer,
    V <: AbstractVocabulary{String},
    C, A, EP, OP, DP, TP
} <: AbstractTransformerTextEncoder
    tokenizer::T
    vocab::V
    config::C
    annotate::A
    process::EP
    onehot::OP
    decode::DP
    textprocess::TP
end

TrfTextEncoder(builder, e::TrfTextEncoder) = TrfTextEncoder(
    getfield(e, :tokenizer), getfield(e, :vocab), getfield(e, :config),
    getfield(e, :annotate),
    builder(e),
    getfield(e, :onehot),
    getfield(e, :decode),
    getfield(e, :textprocess)
)

function Base.getproperty(e::TrfTextEncoder, sym::Symbol)
    if hasfield(TrfTextEncoder, sym)
        return getfield(e, sym)
    else
        return getfield(e, :config)[sym]
    end
end

_membercall(f, e, x) = !(f isa Pipelines) && static_hasmethod(f, Tuple{typeof(e), typeof(x)}) ? f(e, x) : f(x)

TextEncodeBase.tokenize(e::TrfTextEncoder, x) = getfield(e, :tokenizer)(_membercall(getfield(e, :annotate), e, x))
TextEncodeBase.lookup(e::TrfTextEncoder, x) = _membercall(getfield(e, :onehot), e, x)
TextEncodeBase.decode(e::TrfTextEncoder, x) = _membercall(getfield(e, :decode), e, TextEncodeBase.decode_indices(e, x))
TextEncodeBase.decode_text(e::TrfTextEncoder, x) = _membercall(getfield(e, :textprocess), e, TextEncodeBase.decode(e, x))

TextEncodeBase.decode_indices(e::TrfTextEncoder, x) = decode_indices(e, x)
decode_indices(e::TrfTextEncoder, i::Union{Integer, OneHotArray, AbstractArray{<:Integer}}) =
    lookup(String, getfield(e, :vocab), i)
function decode_indices(e::TrfTextEncoder, x::AbstractArray)
    if ndims(x) < 2
        i = argmax(x)
    else
        amax = reshape(argmax(x; dims=1), Base.tail(size(x)))
        i = selectdim(reinterpret(reshape, Int, amax), 1, 1)
    end
    return decode_indices(e, i)
end

annotate_strings(x::AbstractString) = Sentence(x)
annotate_strings(x::Vector{<:AbstractString}) = Batch{Sentence}(x)
annotate_strings(x::Vector{<:Vector{<:AbstractString}}) = Batch{Batch{Sentence}}(x)
annotate_strings(x::Vector{<:Vector{<:Vector{<:AbstractString}}}) = Batch{Batch{Batch{Sentence}}}(x)

TextEncodeBase.onehot_encode(e::TrfTextEncoder, x) = lookup(OneHot, getfield(e, :vocab), x)

lookup_first(e::TrfTextEncoder, x) = TextEncodeBase.onehot_encode(e, x)
lookup_first(e::TrfTextEncoder, x::Tuple) = (TextEncodeBase.onehot_encode(e, x[1]), Base.tail(x)...)
function lookup_first(e::TrfTextEncoder, x::NamedTuple{name}) where name
    xt = Tuple(x)
    return NamedTuple{name}((TextEncodeBase.onehot_encode(e, xt[1]), Base.tail(xt)...))
end

# encoder constructor

const WList = Union{AbstractVector, AbstractVocabulary}

function TransformerTextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary{String}, process,
                                startsym::String, endsym::String, padsym::String, trunc::Union{Nothing, Int})
    return TrfTextEncoder(
        tkr, vocab,
        @NamedTuple{startsym::String, endsym::String, padsym::String, trunc::Union{Nothing, Int}}(
            (startsym, endsym, padsym, trunc)),
        annotate_strings,
        process,
        lookup_first,
        identity,
        TextEncodeBase.join_text,
    )
end

TransformerTextEncoder(tokenizef, v::WList, args...; kws...) =
    TransformerTextEncoder(WordTokenization(tokenize=tokenizef), v, args...; kws...)
TransformerTextEncoder(v::WList, args...; kws...) = TransformerTextEncoder(TextTokenizer(), v, args...; kws...)
TransformerTextEncoder(t::AbstractTokenization, v::WList, args...; kws...) =
    TransformerTextEncoder(TextTokenizer(t), v, args...; kws...)

TransformerTextEncoder(tkr::AbstractTokenizer, v::WList, args...; kws...) =
    throw(MethodError(TransformerTextEncoder, (tkr, v, args...)))

function TransformerTextEncoder(tkr::AbstractTokenizer, words::AbstractVector, process; trunc = nothing,
                                startsym = "<s>", endsym = "</s>", unksym = "<unk>", padsym = "<pad>")
    vocab_list = copy(words)
    for sym in (padsym, unksym, startsym, endsym)
        sym ∉ vocab_list && push!(vocab_list, sym)
    end
    vocab = Vocab(vocab_list, unksym)
    return TransformerTextEncoder(tkr, vocab, process, startsym, endsym, padsym, trunc)
end

function TransformerTextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary, process; trunc = nothing,
                                startsym = "<s>", endsym = "</s>", unksym = "<unk>", padsym = "<pad>")
    check_vocab(vocab, startsym) || @warn "startsym $startsym not in vocabulary, this might cause problem."
    check_vocab(vocab, endsym) || @warn "endsym $endsym not in vocabulary, this might cause problem."
    check_vocab(vocab, unksym) || @warn "unksym $unksym not in vocabulary, this might cause problem."
    check_vocab(vocab, padsym) || @warn "padsym $padsym not in vocabulary, this might cause problem."
    return TransformerTextEncoder(tkr, vocab, process, startsym, endsym, padsym, trunc)
end

function TransformerTextEncoder(tkr::AbstractTokenizer, v::WList;
                                fixedsize = false, trunc_end = :tail, pad_end = :tail, kws...)
    enc = TransformerTextEncoder(tkr, v, identity; kws...)
    # default processing pipeline
    return TransformerTextEncoder(enc) do e
        truncf = get_trunc_pad_func(e.padsym, fixedsize, e.trunc, trunc_end, pad_end)
        maskf = get_mask_func(e.trunc, :tail)
        # get token and convert to string
        Pipeline{:token}(nestedcall(string_getvalue), 1) |>
            # add start & end symbol
            Pipeline{:token}(with_head_tail(e.startsym, e.endsym), :token) |>
            # get mask with specific length
            Pipeline{:attention_mask}(maskf, :token) |>
            # truncate input that exceed length limit and pad them to have equal length
            Pipeline{:token}(truncf, :token) |>
            # convert to dense array
            Pipeline{:token}(nested2batch, :token) |>
            # return token and mask
            PipeGet{(:token, :attention_mask)}()
    end
end

TransformerTextEncoder(builder, e::TrfTextEncoder) = TrfTextEncoder(builder, e)

## encoder-decoder encoding
function TextEncodeBase.encode(e::AbstractTransformerTextEncoder, src, trg)
    psrc = encode(e, src)
    ptrg = encode(e, trg)
    cross_attention_mask = AttenMask(ptrg.attention_mask, psrc.attention_mask)
    return (encoder_input = psrc, decoder_input = merge(ptrg, (; cross_attention_mask)))
end

include("bert_textencoder.jl")
include("gpt_textencoder.jl")
include("t5_textencoder.jl")

end
