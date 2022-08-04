using ..Transformers.BidirectionalEncoder
using ..Transformers.BidirectionalEncoder: WordPiece, WordPieceTokenization,
    BertUnCasedPreTokenization, BertCasedPreTokenization

tokenizer_type(T::Val{:bert}) = T

function load_tokenizer(::Val{:bert}, model_name; force_fast_tkr = false,
                        possible_files = nothing, config = nothing, tkr_cfg = nothing, kw...)
    possible_files = ensure_possible_files(possible_files, model_name; kw...)
    config = ensure_config(config, model_name; kw...)

    if isnothing(tkr_cfg) && TOKENIZER_CONFIG_FILE in possible_files
        tkr_cfg = load_tokenizer_config(model_name; kw...)
    end

    if !isnothing(tkr_cfg)
        model_max_length = get(tkr_cfg, :model_max_length, config.max_position_embeddings)
        lower = get(tkr_cfg, :do_lower_case, true)
        unk_token = get(tkr_cfg, :unk_token, "[UNK]")
        cls_token = get(tkr_cfg, :cls_token, "[CLS]")
        sep_token = get(tkr_cfg, :sep_token, "[SEP]")
        pad_token = get(tkr_cfg, :pad_token, "[PAD]")
    else
        model_max_length = config.max_position_embeddings
        lower = true
        unk_token = "[UNK]"
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        pad_token = "[PAD]"
    end

    if SPECIAL_TOKENS_MAP_FILE in possible_files
        special_tokens = json_load(hgf_tokenizer_special_tokens_map(model_name; kw...))
        match_tokens = collect(values(special_tokens))
        cls_token = get(special_tokens, :cls_token, cls_token)
        sep_token = get(special_tokens, :sep_token, sep_token)
        unk_token = get(special_tokens, :unk_token, unk_token)
        pad_token = get(special_tokens, :pad_token, pad_token)
    else
        special_tokens = nothing
        match_tokens = nothing
    end

    if !force_fast_tkr && "vocab.txt" in possible_files
        tkr_dict = nothing
        vocab_list = readlines(hgf_vocab(model_name; kw...))
        max_char = 200
    elseif FULL_TOKENIZER_FILE in possible_files
        tkr_dict = json_load(hgf_tokenizer(model_name; kw...))
        wp_dict = tkr_dict[:model]
        vocab_list = reverse_keymap_to_list(wp_dict[:vocab])
        unk_token = wp_dict[:unk_token]
        max_char = wp_dict[:max_input_chars_per_word]
    else
        error("Cannot not find vocab.txt or $FULL_TOKENIZER_FILE in $model_name repo")
    end

    if ADDED_TOKENS_FILE in possible_files
        added_tokens = load_tokenizer_added_tokens(model_name; kw...)
        if !isempty(added_tokens)
            for (idx, token) in sort!(collect(Iterators.map(reverse, added_tokens)))
                n_vocab = length(vocab_list)
                if n_vocab == idx
                    push!(vocab_list, token)
                elseif n_vocab > idx
                    idx += 1
                    @assert vocab_list[idx] == token "Two word has same index: $(token) and $(vocab_list[idx])"
                else
                    error("There is a gap in the vocabulary")
                end
            end

            if isnothing(match_tokens)
                match_tokens = collect(keys(added_tokens))
            else
                union!(match_tokens, keys(added_tokens))
            end
        end
    end

    if !isnothing(tkr_dict)
        for dict in sort(tkr_dict[:added_tokens]; by = Base.Fix2(getindex, :id))
            idx = dict[:id]
            token = dict[:content]

            (dict[:rstrip] || dict[:lstrip]) &&
                tokenizer_warn("match token `$token` require to match with space on either side but that is not implemented here")
            dict[:single_word] &&
                tokenizer_warn("match token `$token` does not match inside of a word but that is not implemented here")

            if dict[:special]
                @assert idx <= length(vocab_list)
                @assert vocab_list[idx + 1] == token
            else
                n_vocab = length(vocab_list)
                if n_vocab == idx
                    push!(vocab_list, token)
                elseif n_vocab > idx
                    idx += 1
                    @assert vocab_list[idx] == token "Two word has same index: $(token) and $(vocab_list[idx])"
                else
                    error("There is a gap in the vocabulary")
                end
            end

            if isnothing(match_tokens)
                match_tokens = String[token]
            else
                union!(match_tokens, (token,))
            end
        end
    end
    isnothing(match_tokens) || @assert all(Base.Fix2(in, vocab_list), match_tokens) match_tokens
    wordpiece = WordPiece(vocab_list, unk_token; max_char)

    kwargs = Dict{Symbol, Any}(
        :trunc=>model_max_length, :startsym=>cls_token, :endsym=>sep_token, :padsym=>pad_token
    )

    if !isnothing(tkr_dict)
        norm_dict = tkr_dict[:normalizer]
        @assert norm_dict[:type] == "BertNormalizer"
        lower = norm_dict[:lowercase]
        !isnothing(norm_dict[:strip_accents]) && lower != norm_dict[:strip_accents] &&
            tokenizer_warn("strip_accents and lowercase are not aligned in the loaded tokenizer")
        @assert tkr_dict[:pre_tokenizer][:type] == "BertPreTokenizer"
        trunc = nothing
        if !isnothing(tkr_dict[:padding])
            @assert get(tkr_dict[:padding], :direction, "Right") == "Right" "Cannot padding on left"
            kwargs[:fixedsize] = true
            trunc = get(tkr_dict[:padding], :max_length, trunc)
            kwargs[:padsym] = get(tkr_dict[:padding], :pad_token, pad_token)
        end
        if !isnothing(tkr_dict[:truncation])
            strategy = get(tkr_dict[:truncation], :strategy, nothing)
            !isnothing(strategy) && strategy != "LongestFirst" &&
                tokenizer_warn("truncation strategy $strategy not support, only LongestFirst")
            get(tkr_dict[:truncation], :stride, 0) != 0 &&
                tokenizer_warn("truncation stride is not 0")
            @assert get(tkr_dict[:truncation], :direction, "Right") == "Right" "Cannot truncate on left"
            trunc = get(tkr_dict[:truncation], :max_length, trunc)
        end
        kwargs[:trunc] = trunc
    end

    tkz = lower ? BertUnCasedPreTokenization() : BertCasedPreTokenization()
    tkz = WordPieceTokenization(tkz, wordpiece)
    return BertTextEncoder(tkz; match_tokens, kwargs...)
end
