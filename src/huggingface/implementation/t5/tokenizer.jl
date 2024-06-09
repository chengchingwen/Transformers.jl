using ..TextEncoders: T5TextEncoder

tokenizer_type(T::Val{:t5}) = T
encoder_construct(::Val{:t5}, tokenizer, vocab; kwargs...) = T5TextEncoder(tokenizer, vocab; kwargs...)

function extract_fast_tkr_kwargs(::Val{:t5}, config, special_tokens; eos_token = "</s>", pad_token = "<pad>", kw...)
    if !isnothing(special_tokens)
        eos_token = get(special_tokens, :eos_token, eos_token)
        pad_token = get(special_tokens, :pad_token, pad_token)
    end
    kwargs = Dict{Symbol, Any}()
    kwargs[:endsym] = eos_token
    kwargs[:padsym] = pad_token
    return kwargs
end
