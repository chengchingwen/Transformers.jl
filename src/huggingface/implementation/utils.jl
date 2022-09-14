using ValSplit
using HuggingFaceApi: list_model_files
using StructWalk: scan
using TextEncodeBase
using TextEncodeBase: CodeNormalizer, ReplaceNormalizer,
    MatchTokenization, EachSplitTokenization, EachMatchTokenization, TokenizerStyle
using ..BidirectionalEncoder: WordPiece, BertUnCasedPreTokenization, BertCasedPreTokenization, WordPieceTokenization
using BytePairEncoding
using BytePairEncoding: GPT2Tokenization, gpt2_codemap
using ..Basic: TextTokenizer

ensure_possible_files(possible_files, model_name; revision = nothing, auth_token = nothing, kw...) =
    _ensure(list_model_files, possible_files, model_name; revision, token = auth_token)

ensure_config(config, model_name; kw...) = _ensure(load_config, config, model_name; kw...)

function reverse_keymap_to_list(dict)
    vocab_list = Vector{String}(undef, length(dict))
    for (k, v) in dict
        v += 1
        @assert !isassigned(vocab_list, v) "Two word has same index: $(k) and $(vocab_list[v])"
        vocab_list[v] = k
    end
    @assert all(Base.Fix1(isassigned, vocab_list), eachindex(vocab_list)) "There is a gap in the vocabulary"
    return vocab_list
end

tokenizer_warn(msg) = @warn "$msg, the tokenization result might be slightly different in some cases."

load_error_msg(msg) = "$msg\nFile an issue with the model name you want to load."
load_error(msg) = error(load_error_msg(msg))

function rank_from_lines(lines)
    rank = Dict{NTuple{2, BytePairEncoding.Merge}, Int}()
    for (i, line) in enumerate(lines)
        p = BytePairEncoding.parse_merge(line, nothing)
        rank[p] = i
    end
    return rank
end

empty_then_nothing(::Nothing) = nothing
empty_then_nothing(x) = isempty(unique!(x)) ? nothing : x

function load_special_tokens_map(special_tokens_map_json)
    special_tokens = Dict{Symbol, String}()
    for (k, v) in json_load(special_tokens_map_json)
        special_tokens[k] = v isa String ? v : v["content"]
    end
    return special_tokens
end

function load_added_tokens(added_tokens_json, vocab_list)
    added_tokens = String[]
    added_tokens_dict = JSON.parsefile(added_tokens_json)
    for (idx, token) in sort!(collect(Iterators.map(reverse, added_tokens_dict)))
        n_vocab = length(vocab_list)
        if n_vocab == idx
            push!(vocab_list, token)
        elseif n_vocab > idx
            idx += 1
            @assert vocab_list[idx] == token "Two word has same index: $(token) and $(vocab_list[idx])"
        else
            error("There is a gap in the vocabulary")
        end
        push!(added_tokens, token)
    end
    return isempty(added_tokens) ? nothing :  added_tokens
end

function load_and_add_tokens(added_tokens_file, vocab_list, special_tokens)
    match_tokens = isnothing(added_tokens_file) ? nothing : load_added_tokens(added_tokens_file, vocab_list)
    if !isnothing(special_tokens)
        special_token_values = values(special_tokens)
        match_tokens = isnothing(match_tokens) ?
            collect(special_token_values) : append!(match_tokens, special_token_values)
    end
    match_tokens = empty_then_nothing(match_tokens)
    return match_tokens
end

function extract_added_token(added_token)
    vidx = added_token["id"] + 1
    token = added_token["content"]
    isspecial = added_token["special"]

    added_token["rstrip"] || added_token["lstrip"] && tokenizer_warn(
        "match token `$token` require to match with space on either side but that is not implemented here"
    )
    added_token["single_word"] && tokenizer_warn(
        "match token `$token` does not match inside of a word but that is not implemented here"
    )
    return vidx, token, isspecial
end

extract_and_add_tokens!(::Nothing, _) = nothing
function extract_and_add_tokens!(added_token_list, vocab_list)
    iszero(length(added_token_list)) && return nothing
    sort!(added_token_list; by = Base.Fix2(getindex, "id"))
    match_tokens = String[]
    for added_token in added_token_list
        vidx, token, isspecial = extract_added_token(added_token)
        if isspecial
            # special tokens should be in the vocab already
            @assert vidx <= length(vocab_list)
            @assert vocab_list[vidx] == token
        else
            n_vocab = length(vocab_list)
            if vidx == n_vocab + 1
                push!(vocab_list, token)
            elseif vidx <= n_vocab
                @assert vocab_list[vidx] == token "Two word has same index: $(token) and $(vocab_list[idx])"
            else
                error("There is a gap in the vocabulary")
            end
        end
        push!(match_tokens, token)
    end
    return match_tokens
end

function guessing_tokenization_method(model_dict)
    # method should be one of WordLevel, WordPiece, BPE, Unigram
    if haskey(model_dict, "type")
        # if serialization object has "type" field, use that value directly.
        return Symbol(model_dict["type"])
    elseif haskey(model_dict, "merges")
        # bpe method must have "merges"
        # https://github.com/huggingface/tokenizers/blob/06025e4ca151dcc7f6a4872e8857e8f175e8d3ac/tokenizers/src/models/bpe/serialization.rs#L59
        return :BPE
    elseif haskey(model_dict, "max_input_chars_per_word") || haskey(model_dict, "continuing_subword_prefix")
        # only bpe and wordpiece has "continuing_subword_prefix", so if it doesn't has "merges" but has
        # "continuing_subword_prefix", it should be wordpiece.
        # https://github.com/huggingface/tokenizers/blob/06025e4ca151dcc7f6a4872e8857e8f175e8d3ac/tokenizers/src/models/wordpiece/serialization.rs#L40-L41
        return :WordPiece
    elseif haskey(model_dict, "unk_id")
        # unigram only have "vocab" and "unk_id", but it seems only unigram use "unk_id" (others use "unk_token").
        # https://github.com/huggingface/tokenizers/blob/06025e4ca151dcc7f6a4872e8857e8f175e8d3ac/tokenizers/src/models/unigram/serialization.rs#L28
        return :Unigram
    elseif haskey(model_dict, "unk_token")
        # wordlevel only have "vocab" and "unk_token". At this point, it's probably wordlevel, but it should at least
        # have "unk_token".
        # https://github.com/huggingface/tokenizers/blob/06025e4ca151dcc7f6a4872e8857e8f175e8d3ac/tokenizers/src/models/wordlevel/serialization.rs#L30
        return :WordLevel
    else
        # Otherwise we raise an error.
        error("Failed to guess the tokenization method")
    end
end

@valsplit extract_tokenization_method(Val(method::Symbol), model_dict) =
    load_error("Unsupported tokenization method: $method")

extract_tokenization_method(model_dict) = extract_tokenization_method(
    guessing_tokenization_method(model_dict), model_dict)

function extract_tokenization_method(::Val{:WordPiece}, model_dict)
    @assert model_dict["continuing_subword_prefix"] == "##"
    unk_token = model_dict["unk_token"]
    max_char = model_dict["max_input_chars_per_word"]
    vocab_list = reverse_keymap_to_list(model_dict["vocab"])
    wordpiece = WordPiece(vocab_list, unk_token; max_char)
    return Base.Fix2(WordPieceTokenization, wordpiece), wordpiece, unk_token, vocab_list
end

empty2nothing(::Nothing) = nothing
empty2nothing(s) = isempty(s) ? nothing : s

function extract_tokenization_method(::Val{:BPE}, model_dict)
    @assert isnothing(model_dict["dropout"]) "BPE with dropout unsupported"
    @assert !model_dict["fuse_unk"] "fuse_unk is unsupported"
    unk_token = model_dict["unk_token"]
    sepsym = empty2nothing(model_dict["continuing_subword_prefix"])
    endsym = empty2nothing(model_dict["end_of_word_suffix"])
    merges = rank_from_lines(model_dict["merges"])
    bpe = CachedBPE(BPE(merges, sepsym, endsym))
    vocab_list = reverse_keymap_to_list(model_dict["vocab"])
    return Base.Fix2(BPETokenization, bpe), bpe, unk_token, vocab_list
end

# extract_tokenization_method(::Val{:Unigram}, model_dict)
# extract_tokenization_method(M::Val{:WordLevel}, model_dict)

function extract_tokenizer_model(model_dict)
    method, object, unk, vocab_list = extract_tokenization_method(model_dict)
    return method, object, unk, vocab_list
end

@valsplit extract_pre_tokenization(
    Val(tokenization_type::Symbol), pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict
) = load_error("Unsupported tokenization method: $tokenization_type")

extract_pre_tokenization(pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict) =
    extract_pre_tokenization(
        Symbol(pretokenizer_dict["type"]), pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict)

function extract_pre_tokenization(
    ::Val{:BertPreTokenizer}, pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict
)
    @assert isnothing(tokenization) load_error_msg("Chaining tokenization is unsupported")
    normalizer_dict = tokenizer_dict["normalizer"]
    @assert !isnothing(normalizer_dict) && normalizer_dict["type"] == "BertNormalizer" load_error_msg("Normalizer of BertPreTokenizer is not BertNormalizer")
    islower = normalizer_dict["lowercase"]
    pretokenization = islower ? BertUnCasedPreTokenization() : BertCasedPreTokenization()
    return pretokenization, match_tokens, normalizer
end

function extract_pre_tokenization(
    ::Val{:ByteLevel}, pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict
)
    @assert !pretokenizer_dict["add_prefix_space"] "add_prefix_space is unsupported"
    @assert pretokenizer_dict["trim_offsets"]
    isnothing(tokenization) && (tokenization = GPT2Tokenization())
    normalizer = normalizer ∘ Base.Fix2(CodeNormalizer, gpt2_codemap())
    return tokenization, match_tokens, normalizer
end

function extract_pre_tokenization(
    ::Val{:Split}, pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict
)
    @assert isnothing(tokenization) load_error_msg("Chaining tokenization is unsupported")
    @assert pretokenizer_dict["behavior"] == "Removed" load_error_msg("Only support removed behavior")
    @assert haskey(pretokenizer_dict["pattern"], "Regex") load_error_msg("Only support regex pattern")
    regex = Regex(pretokenizer_dict["pattern"]["Regex"])
    if pretokenizer_dict["invert"]
        if regex == r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
            tokenization = GPT2Tokenization()
        else
            if regex == r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"
                regex = r"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"
                if isnothing(match_tokens)
                    match_tokens = ["<|startoftext|>", "<|endoftext|>"]
                else
                    push!(match_tokens, "<|startoftext|>", "<|endoftext|>")
                end
            end
            tokenization = EachMatchTokenization(regex)
        end
    else
        tokenization = EachSplitTokenization(regex)
    end
    return tokenization, match_tokens, normalizer
end

function extract_pre_tokenization(
    ::Val{:WhitespaceSplit}, pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict
)
    @assert isnothing(tokenization) load_error_msg("Chaining tokenization is unsupported")
    return EachSplitTokenization(isspace), match_tokens, normalizer
end

function extract_pre_tokenization(
    ::Val{:Whitespace}, pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict
)
    @assert isnothing(tokenization) load_error_msg("Chaining tokenization is unsupported")
    return EachMatchTokenization(r"\w+|[^\w\s]+"), match_tokens, normalizer
end

function extract_pre_tokenization(
    ::Val{:Sequence}, pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict
)
    for sub_pretokenizer_dict in pretokenizer_dict["pretokenizers"]
        tokenization, match_tokens, normalizer = extract_pre_tokenization(
            sub_pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict)
    end
    return tokenization, match_tokens, normalizer
end

function extract_base_tokenization(method, match_tokens, tokenizer_dict)
    # Rust fast tokenizer works in the order of: normalization -> pre-tokenization -> tokenization method
    # However, TextEncodeBase tokenization struct basically wrapped in the reverse way, i.e. the
    # normalization should be the outter-most wrapper. OTOH, our tokenization methods is usually defined
    # as a wrapper struct, which means our overall struct would be in the order of:
    # normalization(tokenization method(pre-tokenization))
    pretokenization, match_tokens, normalizer = extract_pre_tokenization(
        tokenizer_dict["pre_tokenizer"], nothing, match_tokens, identity, tokenizer_dict)
    tokenization = normalizer(method(pretokenization))
    base_tokenization = extract_normalizer(tokenizer_dict["normalizer"], tokenization, tokenizer_dict)
    return base_tokenization, match_tokens
end

extract_normalizer(::Nothing, tokenization, tokenizer_dict) = tokenization
extract_normalizer(normalizer_dict, tokenization, tokenizer_dict) =
    extract_normalizer(Symbol(normalizer_dict["type"]), normalizer_dict, tokenization, tokenizer_dict)

@valsplit extract_normalizer(Val(normalizer_type::Symbol), normalizer_dict, tokenization, tokenizer_dict) =
    load_error("Unsupported normalizer method: $normalizer_type")

function extract_normalizer(::Val{:BertNormalizer}, normalizer_dict, tokenization, tokenizer_dict)
    # bert normalizer is done in bert pre tokenization
    check = Ref{Bool}(false)
    scan(x -> (x isa BertUnCasedPreTokenization || x isa BertCasedPreTokenization) && (check[] = true),
         TokenizerStyle(), tokenization)
    check[] || load_error("BertNormalizer without BertPreTokenizer is unsupported")
    return tokenization
end

extract_normalizer(::Val{:Lowercase}, normalizer_dict, tokenization, tokenizer_dict) =
    TextEncodeBase.LowercaseNormalizer(tokenization)

extract_normalizer(::Val{:NFD}, normalizer_dict, tokenization, tokenizer_dict) =
    TextEncodeBase.UnicodeNormalizer(tokenization, :NFD)

extract_normalizer(::Val{:NFKD}, normalizer_dict, tokenization, tokenizer_dict) =
    TextEncodeBase.UnicodeNormalizer(tokenization, :NFKD)

extract_normalizer(::Val{:NFC}, normalizer_dict, tokenization, tokenizer_dict) =
    TextEncodeBase.UnicodeNormalizer(tokenization, :NFC)

extract_normalizer(::Val{:NFKC}, normalizer_dict, tokenization, tokenizer_dict) =
    TextEncodeBase.UnicodeNormalizer(tokenization, :NFKC)

function extract_normalizer(::Val{:Replace}, normalizer_dict, tokenization, tokenizer_dict)
    @assert haskey(normalizer_dict["pattern"], "Regex") load_error_msg("Only support regex pattern")
    content = normalizer_dict["content"]
    regex = Regex(normalizer_dict["pattern"]["Regex"])
    return ReplaceNormalizer(tokenization, regex=>content)
end

function extract_normalizer(::Val{:Sequence}, normalizer_dict, tokenization, tokenizer_dict)
    for sub_normalizer_dict in Iterators.reverse(normalizer_dict["normalizers"])
        tokenization = extract_normalizer(sub_normalizer_dict, tokenization, tokenizer_dict)
    end
    return tokenization
end

function extract_trunc_pad(tokenizer_dict)
    # In huggingface's setting, padding and truncation can be applied independently.
    # Therefore, both of them would have a `max_length` fields.
    # But in our setting, they must be applied together (since we are going to convert to tensor).
    # We use the `max_length` of `truncation` as the possible final output length.
    process_config = Dict{Symbol, Any}()
    pad_dict = tokenizer_dict["padding"]
    trunc_dict = tokenizer_dict["truncation"]
    trunc = nothing
    if !isnothing(pad_dict)
        process_config[:fixedsize] = true
        process_config[:pad_end] = get(pad_dict, "direction", "Left") == "Left" ? :head : :tail
        padsym = get(pad_dict, "pad_token", nothing)
        isnothing(padsym) || (process_config[:padsym] = padsym)
        trunc = get(pad_dict, "max_length", trunc)
        isnothing(trunc) || (process_config[:trunc] = trunc)
    end
    if !isnothing(trunc_dict)
        strategy = get(trunc_dict, "strategy", nothing)
        !isnothing(strategy) && strategy != "LongestFirst" &&
            tokenizer_warn("truncation strategy $strategy not support, only LongestFirst")
        get(trunc_dict, "stride", 0) != 0 &&
            tokenizer_warn("truncation stride is not 0")
        process_config[:trunc_end] = get(trunc_dict, "direction", "Left") == "Left" ? :head : :tail
        trunc = get(trunc_dict, "max_length", trunc)
        isnothing(trunc) || (process_config[:trunc] = trunc)
    end
    return process_config
end

function load_fast_tokenizer_components(tokenizer_json)
    tokenizer_dict = JSON.parsefile(tokenizer_json)
    method, tokenization_object, unk, vocab_list = extract_tokenizer_model(tokenizer_dict["model"])
    match_tokens = extract_and_add_tokens!(tokenizer_dict["added_tokens"], vocab_list)
    base_tokenization, match_tokens = extract_base_tokenization(method, match_tokens, tokenizer_dict)
    match_tokens = empty_then_nothing(match_tokens)
    process_config = extract_trunc_pad(tokenizer_dict)
    # we are ignoring the "post_processor" here.
    return base_tokenization, match_tokens, vocab_list, unk, tokenization_object, process_config
end

load_fast_tokenizer(type, tokenizer_json, args...) = load_fast_tokenizer(tokenizer_json) # default ignoring type
function load_fast_tokenizer(tokenizer_json)
    base_tokenization, match_tokens, vocab_list, unk, tokenization_object, process_config =
        load_fast_tokenizer_components(tokenizer_json)
    isnothing(match_tokens) || (base_tokenization = MatchTokenization(base_tokenization, match_tokens))
    isnothing(unk) && (unk = "<unk>") # dummy unk token, wouldn't appear in vocabulary
    unk isa AbstractString || (unk = vocab_list[unk + 1])
    vocab = Vocab(vocab_list, unk)
    tokenizer = TextTokenizer(base_tokenization)
    return tokenizer, vocab, process_config
end

@valsplit load_slow_tokenizer(Val(type::Symbol), args...; kwargs...) = error("No slow tokenizer defined for $type.")

extract_tkr_kwargs(type, ::Nothing, config, special_tokens) = extract_tkr_kwargs(type, config, special_tokens)
extract_tkr_kwargs(type, tkr_cfg, config, special_tokens) = extract_tkr_kwargs(type, config, special_tokens; tkr_cfg...)
@valsplit extract_tkr_kwargs(Val(type::Symbol), config, special_tokens; kw...) =
    error("Don't know how to extract arguments for $type tokenizer.")

@valsplit slow_tkr_files(Val(type::Symbol)) = error("Don't know what files are need to load slow $type tokenizer.")
@valsplit encoder_construct(Val(type::Symbol), tokenizer, vocab; kwargs...) = error("Don't know how to construct text encoder for $type")
