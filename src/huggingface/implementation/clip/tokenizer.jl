using ..TextEncoders: GPT2TextEncoder
using BytePairEncoding
using TextEncodeBase

tokenizer_type(T::Val{:clip}) = T
encoder_construct(::Val{:clip}, tokenizer, vocab; kwargs...) = encoder_construct(Val{:gpt2}(), tokenizer, vocab; kwargs...)
slow_tkr_files(::Val{:clip}) = slow_tkr_files(Val{:gpt2}())

function extract_fast_tkr_kwargs(
    ::Val{:clip}, config, special_tokens;
    bos_token = "<|startoftext|>", eos_token = "<|endoftext|>", pad_token = "<|endoftext|>",
    model_max_length = nothing, kw...
)
    text_config = config isa HGFConfig{:clip_text} ? config : get_key(config, :text_config, nothing)
    if isnothing(model_max_length)
        model_max_length = isnothing(text_config) ? 77 : get_key(text_config, :max_position_embeddings, 77)
    end
    if !isnothing(special_tokens)
        unk_token = get(special_tokens, :unk_token, unk_token)
        bos_token = get(special_tokens, :bos_token, bos_token)
        eos_token = get(special_tokens, :eos_token, eos_token)
        pad_token = get(special_tokens, :pad_token, pad_token)
    end
    kwargs = Dict{Symbol, Any}()
    kwargs[:startsym] = bos_token
    kwargs[:endsym] = eos_token
    kwargs[:padsym] = pad_token
    kwargs[:trunc] = model_max_length
    return kwargs
end

function extract_slow_tkr_kwargs(::Val{:clip}, config, special_tokens; unk_token = "<|endoftext|>", kw...)
    if !isnothing(special_tokens)
        unk_token = get(special_tokens, :unk_token, unk_token)
    end
    slow_tkr_kwargs = Dict{Symbol, Any}()
    slow_tkr_kwargs[:unk_token] = unk_token
    return slow_tkr_kwargs
end
