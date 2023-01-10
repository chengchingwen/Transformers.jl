using ..TextEncoders: GPT2TextEncoder
using BytePairEncoding
using BytePairEncoding: CachedBPE, GPT2Tokenization, gpt2_codemap
using TextEncodeBase

tokenizer_type(T::Val{:gpt2}) = T
encoder_construct(::Val{:gpt2}, tokenizer, vocab; kwargs...) = GPT2TextEncoder(tokenizer, vocab; kwargs...)
slow_tkr_files(::Val{:gpt2}) = (VOCAB_JSON_FILE, MERGES_FILE)

function load_slow_tokenizer(
    ::Val{:gpt2}, vocab_file, merges_file, added_tokens_file = nothing, special_tokens = nothing;
    unk_token = "<|endoftext|>"
)
    vocab_list = reverse_keymap_to_list(json_load(vocab_file))
    bpe = CachedBPE(BPE(merges_file))
    match_tokens = load_and_add_tokens(added_tokens_file, vocab_list, special_tokens)
    base_tokenization = BPETokenization(GPT2Tokenization(), bpe)
    base_tokenization = CodeNormalizer(base_tokenization, gpt2_codemap())
    isnothing(match_tokens) || (base_tokenization = MatchTokenization(base_tokenization, match_tokens))
    tokenizer = TextTokenizer(base_tokenization)
    return tokenizer, Vocab(vocab_list, unk_token), (;)
end

function extract_fast_tkr_kwargs(
    ::Val{:gpt2}, config, special_tokens;
    bos_token = "<|endoftext|>", eos_token = "<|endoftext|>", pad_token = "<|endoftext|>",
    model_max_length = get(config, :n_positions, 1024), kw...
)
    if !isnothing(special_tokens)
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

function extract_slow_tkr_kwargs(::Val{:gpt2}, config, special_tokens; unk_token = "<|endoftext|>", kw...)
    if !isnothing(special_tokens)
        unk_token = get(special_tokens, :unk_token, unk_token)
    end
    slow_tkr_kwargs = Dict{Symbol, Any}()
    slow_tkr_kwargs[:unk_token] = unk_token
    return slow_tkr_kwargs
end
