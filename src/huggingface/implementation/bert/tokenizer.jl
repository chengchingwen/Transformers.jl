using ..Transformers.BidirectionalEncoder
using ..Transformers.BidirectionalEncoder: WordPiece, WordPieceTokenization,
    BertUnCasedPreTokenization, BertCasedPreTokenization

tokenizer_type(T::Val{:bert}) = T

function load_slow_tokenizer(::Val{:bert}, vocab_file, added_tokens_file = nothing, special_tokens = nothing;
                             unk_token = "[UNK]", max_char = 200, lower = true)
    vocab_list = readlines(vocab_file)
    match_tokens = load_and_add_tokens(added_tokens_file, vocab_list, special_tokens)
    wordpiece = WordPiece(vocab_list, unk_token; max_char)
    base_tokenization = lower ? BertUnCasedPreTokenization() : BertCasedPreTokenization()
    base_tokenization = WordPieceTokenization(base_tokenization, wordpiece)
    isnothing(match_tokens) || (base_tokenization = MatchTokenization(base_tokenization, match_tokens))
    tokenizer = TextTokenizer(base_tokenization)
    return tokenizer, Vocab(wordpiece), (;)
end

bert_kwargs(::Nothing, config, special_tokens) = bert_kwargs(config, special_tokens)
bert_kwargs(tkr_cfg, config, special_tokens) = bert_kwargs(config, special_tokens; tkr_cfg...)
function bert_kwargs(
    config, special_tokens;
    unk_token = "[UNK]", cls_token = "[CLS]", sep_token = "[SEP]", pad_token = "[PAD]",
    do_lower_case = true, model_max_length = config.max_position_embeddings, kw...
)
    if !isnothing(special_tokens)
        unk_token = get(special_tokens, :unk_token, unk_token)
        cls_token = get(special_tokens, :cls_token, cls_token)
        sep_token = get(special_tokens, :sep_token, sep_token)
        pad_token = get(special_tokens, :pad_token, pad_token)
    end

    kwargs = Dict{Symbol, Any}()
    kwargs[:startsym] = cls_token
    kwargs[:endsym] = sep_token
    kwargs[:padsym] = pad_token
    kwargs[:trunc] = model_max_length

    return kwargs, unk_token, do_lower_case
end

function load_tokenizer(
    T::Val{:bert}, model_name; force_fast_tkr = false, possible_files = nothing,
    config = nothing, tkr_config = nothing,
    kw...
)
    possible_files = ensure_possible_files(possible_files, model_name; kw...)
    config = ensure_config(config, model_name; kw...)

    isnothing(tkr_config) && TOKENIZER_CONFIG_FILE in possible_files &&
        (tkr_config = load_tokenizer_config(model_name; kw...))
    special_tokens = SPECIAL_TOKENS_MAP_FILE in possible_files ?
        load_special_tokens_map(hgf_tokenizer_special_tokens_map(model_name; kw...)) : nothing
    kwargs, unk_token, lower = bert_kwargs(tkr_config, config, special_tokens)

    if FULL_TOKENIZER_FILE in possible_files || force_fast_tkr
        @assert FULL_TOKENIZER_FILE in possible_files "Forcely using fast tokenizer but cannot find $FULL_TOKENIZER_FILE in $model_name repo"
        tokenizer, vocab, process_config = load_fast_tokenizer(T, hgf_tokenizer(model_name; kw...))
    else
        @assert VOCAB_FILE in possible_files "Cannot not find $VOCAB_FILE or $FULL_TOKENIZER_FILE in $model_name repo"
        vocab_file = hgf_vocab(model_name; kw...)
        added_tokens_file = ADDED_TOKENS_FILE in possible_files ? hgf_tokenizer_added_token(model_name; kw...) : nothing
        tokenizer, vocab, process_config = load_slow_tokenizer(
            T, vocab_file, added_tokens_file, special_tokens;
            unk_token, lower
        )
    end

    for (k, v) in process_config
        kwargs[k] = v
    end

    return BertTextEncoder(tokenizer, vocab; kwargs...)
end
