using TextEncodeBase
using TextEncodeBase: trunc_and_pad, trunc_or_pad, nested2batch, nestedcall, Batch, Sentence
using TextEncodeBase: SequenceTemplate, ConstTerm, InputTerm, RepeatedTerm
using FuncPipelines
using ..Basic: string_getvalue, grouping_sentence, check_vocab, TextTokenizer, AbstractTransformerTextEncoder, getmask

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
    if fixedsize
        @assert !isnothing(trunc) "`fixedsize=true` but `trunc` is not set."
        truncf = trunc_or_pad
    else
        truncf = trunc_and_pad
    end

    if isnothing(process)
        process =
            # group input for SequenceTemplate
            Pipeline{:tok}(grouping_sentence, :tok) |>
            # add start & end symbol, compute segment and merge sentences
            Pipeline{:tok}(
                SequenceTemplate(
                    InputTerm{String}(), ConstTerm(endsym),
                    RepeatedTerm(InputTerm{String}(), ConstTerm(endsym, 2)))(Val(1)), :tok)
    end

    # get token and convert to string
    return Pipeline{:tok}(nestedcall(string_getvalue), 1) |>
        process |>
        # truncate input that exceed length limit and pad them to have equal length
        Pipeline{:trunc_tok}(truncf(trunc, padsym, trunc_end, pad_end), :tok) |>
        # get the truncated length
        (fixedsize ?
         PipeVar{:trunc_len}(trunc) :
         Pipeline{:trunc_len}(TextEncodeBase.nestedmaxlength, :trunc_tok)
         ) |>
        # set pad end
        PipeVar{:lpad}(pad_end == :head) |>
        # get mask with specific length
        Pipeline{:mask}(getmask, (:tok, :trunc_len, :lpad)) |>
        # convert to dense array
        Pipeline{:tok}(nested2batch, :trunc_tok) |>
        # input namedtuple
        Pipeline{:input}(NamedTuple{(:tok,)}∘tuple, :tok) |>
        # return input and mask
        PipeGet{(:input, :mask)}()
end

TextEncodeBase.tokenize(e::T5TextEncoder, x::AbstractString) = e.tokenizer(Sentence(x))
TextEncodeBase.tokenize(e::T5TextEncoder, x::Vector{<:AbstractString}) = e.tokenizer(Batch{Sentence}(x))
TextEncodeBase.tokenize(e::T5TextEncoder, x::Vector{<:Vector{<:AbstractString}}) = e.tokenizer(Batch{Batch{Sentence}}(x))

function TextEncodeBase.lookup(e::T5TextEncoder, x::NamedTuple)
    onehot_tok = lookup(e, x.input.tok)
    input = merge(x.input, (tok = onehot_tok,))
    return merge(x, (input = input,))
end


tokenizer_type(T::Val{:t5}) = T
encoder_construct(::Val{:t5}, tokenizer, vocab; kwargs...) = T5TextEncoder(tokenizer, vocab; kwargs...)
# slow_tkr_files(::Val{:t5}) = 

function extract_tkr_kwargs(
    ::Val{:t5}, config, special_tokens;
    unk_token = "<unk>", eos_token = "</s>", pad_token = "<pad>",
    kw...
)
    if !isnothing(special_tokens)
        unk_token = get(special_tokens, :unk_token, unk_token)
        eos_token = get(special_tokens, :eos_token, eos_token)
        pad_token = get(special_tokens, :pad_token, pad_token)
    end

    kwargs = Dict{Symbol, Any}()
    kwargs[:endsym] = eos_token
    kwargs[:padsym] = pad_token

    slow_tkr_kwargs = Dict{Symbol, Any}()
    slow_tkr_kwargs[:unk_token] = unk_token

    return kwargs, slow_tkr_kwargs
end
