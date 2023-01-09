using ..TextEncoders: BertUnCasedPreTokenization, BertCasedPreTokenization, BertTextEncoder
using ..Transformers.WordPieceModel: WordPiece, WordPieceTokenization

tokenizer_type(T::Val{:bert}) = T
encoder_construct(::Val{:bert}, tokenizer, vocab; kwargs...) = BertTextEncoder(tokenizer, vocab; kwargs...)
slow_tkr_files(::Val{:bert}) = (VOCAB_FILE,)

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

function extract_fast_tkr_kwargs(
    ::Val{:bert}, config, special_tokens;
    cls_token = "[CLS]", sep_token = "[SEP]", pad_token = "[PAD]",
    model_max_length = get_key(config, :max_position_embeddings, 512), kw...
)
    if !isnothing(special_tokens)
        cls_token = get(special_tokens, :cls_token, cls_token)
        sep_token = get(special_tokens, :sep_token, sep_token)
        pad_token = get(special_tokens, :pad_token, pad_token)
    end
    kwargs = Dict{Symbol, Any}()
    kwargs[:startsym] = cls_token
    kwargs[:endsym] = sep_token
    kwargs[:padsym] = pad_token
    kwargs[:trunc] = model_max_length
    return kwargs
end

function extract_slow_tkr_kwargs(::Val{:bert}, config, special_tokens; unk_token = "[UNK]", do_lower_case = true, kw...)
    if !isnothing(special_tokens)
        unk_token = get(special_tokens, :unk_token, unk_token)
    end
    slow_tkr_kwargs = Dict{Symbol, Any}()
    slow_tkr_kwargs[:unk_token] = unk_token
    slow_tkr_kwargs[:lower] = do_lower_case
    return slow_tkr_kwargs
end
