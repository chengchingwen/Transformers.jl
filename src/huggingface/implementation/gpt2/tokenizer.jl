using ..Transformers.GenerativePreTrain
using BytePairEncoding
using BytePairEncoding: CachedBPE, GPT2Tokenization, gpt2_codemap
using TextEncodeBase

tokenizer_type(T::Val{:gpt2}) = T

function load_slow_tokenizer(
    ::Val{:gpt2}, vocab_file, merges_file, added_tokens_file = nothing, special_tokens = nothing;
    unk_token
)
    vocab_list = reverse_keymap_to_list(JSON.parsefile(vocab_file))
    bpe = CachedBPE(BPE(merges_file))
    match_tokens = load_and_add_tokens(added_tokens_file, vocab_list, special_tokens)
    base_tokenization = BPETokenization(GPT2Tokenization(), bpe)
    base_tokenization = CodeNormalizer(base_tokenization, gpt2_codemap())
    isnothing(match_tokens) || (base_tokenization = MatchTokenization(base_tokenization, match_tokens))
    tokenizer = TextTokenizer(base_tokenization)
    return tokenizer, Vocab(vocab_list, unk_token), (;)
end

gpt2_kwargs(::Nothing, config, special_tokens) = gpt2_kwargs(config, special_tokens)
gpt2_kwargs(tkr_cfg, config, special_tokens) = gpt2_kwargs(config, special_tokens; tkr_cfg...)
function gpt2_kwargs(
    config, special_tokens;
    unk_token = "<|endoftext|>", bos_token = "<|endoftext|>", eos_token = "<|endoftext|>", pad_token = "<|endoftext|>",
    model_max_length = config.n_positions, kw...
)
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

    return kwargs, unk_token
end


function load_tokenizer(
    T::Val{:gpt2}, model_name; force_fast_tkr = false, possible_files = nothing,
    config = nothing, tkr_config = nothing,
    kw...
)
    possible_files = ensure_possible_files(possible_files, model_name; kw...)
    config = ensure_config(config, model_name; kw...)

    isnothing(tkr_config) && TOKENIZER_CONFIG_FILE in possible_files &&
        (tkr_config = load_tokenizer_config(model_name; kw...))
    special_tokens = SPECIAL_TOKENS_MAP_FILE in possible_files ?
        load_special_tokens_map(hgf_tokenizer_special_tokens_map(model_name; kw...)) : nothing
    kwargs, unk_token = gpt2_kwargs(tkr_config, config, special_tokens)

    if FULL_TOKENIZER_FILE in possible_files || force_fast_tkr
        @assert FULL_TOKENIZER_FILE in possible_files "Forcely using fast tokenizer but cannot find $FULL_TOKENIZER_FILE in $model_name repo"
        tokenizer, vocab, process_config = load_fast_tokenizer(T, hgf_tokenizer(model_name; kw...))
    else
        @assert VOCAB_JSON_FILE in possible_files && MERGES_FILE in possible_files "Cannot not find $VOCAB_FILE and $MERGES_FILE or $FULL_TOKENIZER_FILE in $model_name repo"
        vocab_file = hgf_vocab_json(model_name; kw...)
        merges_file = hgf_merges(model_name; kw...)
        added_tokens_file = ADDED_TOKENS_FILE in possible_files ? hgf_tokenizer_added_token(model_name; kw...) : nothing
        tokenizer, vocab, process_config = load_slow_tokenizer(
            T, vocab_file, merges_file, added_tokens_file, special_tokens;
            unk_token
        )
    end

    for (k, v) in process_config
        kwargs[k] = v
    end

    return GPT2TextEncoder(tokenizer, vocab; kwargs...)
end
