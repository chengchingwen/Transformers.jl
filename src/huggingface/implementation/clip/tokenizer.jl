using ..Transformers.GenerativePreTrain
using BytePairEncoding
using BytePairEncoding: CachedBPE, GPT2Tokenization, gpt2_codemap
using TextEncodeBase
using TextEncodeBase: EachMatchTokenization

tokenizer_type(T::Val{:clip}) = T
encoder_construct(::Val{:clip}, tokenizer, vocab; kwargs...) = encoder_construct(Val{:gpt2}(), tokenizer, vocab; kwargs...)
slow_tkr_files(::Val{:clip}) = slow_tkr_files(Val{:gpt2}())

# function load_slow_tokenizer(
#     ::Val{:clip}, vocab_file, merges_file, added_tokens_file = nothing, special_tokens = nothing;
#     unk_token = "<|endoftext|>"
# )
#     vocab_list = reverse_keymap_to_list(JSON.parsefile(vocab_file))
#     bpe = CachedBPE(BPE(merges_file; endsym = "</w>"))
#     match_tokens = load_and_add_tokens(added_tokens_file, vocab_list, special_tokens)
#     if isnothing(match_tokens)
#         match_tokens = ["<|startoftext|>", "<|endoftext|>"]
#     else
#         push!(match_tokens, "<|startoftext|>", "<|endoftext|>")
#         unique!(match_tokens)
#     end
#     base_tokenization = EachMatchTokenization(r"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+")
#     base_tokenization = BPETokenization(base_tokenization, bpe)
#     base_tokenization = CodeNormalizer(base_tokenization, gpt2_codemap())
#     isnothing(match_tokens) || (base_tokenization = MatchTokenization(base_tokenization, match_tokens))
#     tokenizer = TextTokenizer(base_tokenization)
#     return tokenizer, Vocab(vocab_list, unk_token), (;)
# end

function extract_tkr_kwargs(
    ::Val{:clip}, config, special_tokens;
    unk_token = "<|endoftext|>", bos_token = "<|startoftext|>", eos_token = "<|endoftext|>", pad_token = "<|endoftext|>",
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

    slow_tkr_kwargs = Dict{Symbol, Any}()
    slow_tkr_kwargs[:unk_token] = unk_token

    return kwargs, slow_tkr_kwargs
end
