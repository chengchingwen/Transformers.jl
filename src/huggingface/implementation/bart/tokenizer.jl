using ..TextEncoders: GPT2TextEncoder
using BytePairEncoding
using BytePairEncoding: CachedBPE, GPT2Tokenization, gpt2_codemap
using TextEncodeBase

tokenizer_type(T::Val{:bart}) = T
encoder_construct(::Val{:bart}, tokenizer, vocab; kwargs...) = GPT2TextEncoder(tokenizer, vocab; kwargs...)
slow_tkr_files(::Val{:bart}) = (VOCAB_JSON_FILE, MERGES_FILE)

function extract_fast_tkr_kwargs(
    ::Val{:bart}, config, special_tokens;
    bos_token = "<s>", eos_token = "</s>", pad_token = "<pad>",
    sep_token = "</s>", cls_token = "<s>", mask_token = "<mask>",
    model_max_length = get_key(config, :max_position_embeddings, 1024), kw...
)
    if !isnothing(special_tokens)
        bos_token = get(special_tokens, :bos_token, bos_token)
        eos_token = get(special_tokens, :eos_token, eos_token)
        pad_token = get(special_tokens, :pad_token, pad_token)
        cls_token = get(special_tokens, :cls_token, cls_token)
        sep_token = get(special_tokens, :sep_token, sep_token)
        mask_token = get(special_tokens, :mask_token, mask_token)
    end
    kwargs = Dict{Symbol, Any}()
    kwargs[:startsym] = bos_token
    kwargs[:endsym] = eos_token
    kwargs[:padsym] = pad_token
    kwargs[:trunc] = model_max_length
    return kwargs
end
