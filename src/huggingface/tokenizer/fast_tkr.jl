using StructWalk: scan
using FuncPipelines
using LRUCache
using TextEncodeBase
using TextEncodeBase: CodeNormalizer, ReplaceNormalizer, WordReplaceNormalizer,
    SentenceFuncNormalizer, WordFuncNormalizer,
    MatchTokenization, EachSplitTokenization, EachMatchTokenization, MatchSplitsTokenization,
    TokenizerStyle, nestedcall
using TextEncodeBase: SequenceTemplate, ConstTerm, InputTerm, RepeatedTerm, IndexInputTerm
using TextEncodeBase.RustRegex
using ..TextEncoders: BertUnCasedPreTokenization, BertCasedPreTokenization, TextTokenizer,
    grouping_sentence, string_strip
using ..WordPieceModel
using BytePairEncoding
using BytePairEncoding: CachedBPE, ByteFallbackBPE, GPT2Tokenization, gpt2_codemap, fallback2byte
using ..UnigramLanguageModel
using ..UnigramLanguageModel: PrecompiledNormalizer, CachedUnigram

struct NoTokenization <: TextEncodeBase.BaseTokenization end
TextEncodeBase.splitting(::NoTokenization, s::TextEncodeBase.SentenceStage) = Base.vect(TextEncodeBase.getvalue(s))

# https://github.com/huggingface/transformers/blob/235e5d4991e8a0984aa78db91087b49622c7740e/src/transformers/tokenization_utils_base.py#L3798
# NOT https://github.com/huggingface/tokenizers/blob/daf361676bdfd14088f7e0bc087effc6a9cfdf3e/tokenizers/src/decoders/wordpiece.rs#L31
cleanup(s) = replace(
    replace(replace(s, " ." => ".", " ?" => "?", " !" => "!", " ," => ","), " ' " => "'"),
    " n't" => "n't", " 'm" => "'m", #= " do not" => " don't", =#
    " 's" => "'s", " 've" => "'ve", " 're" => "'re")

add_prefix(prefix) = Base.Fix1(add_prefix, prefix)
add_prefix(prefix, str) = prefix * str
ensure_prefix(prefix) = Base.Fix1(ensure_prefix, prefix)
ensure_prefix(prefix, str) = String(startswith(str, prefix) ? str : add_prefix(prefix, str))

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
    added_token_list = sort(added_token_list; by = Base.Fix2(getindex, "id"))
    match_tokens = String[]
    for added_token in added_token_list
        vidx, token, isspecial = extract_added_token(added_token)
        if isspecial
            if vidx > length(vocab_list)
                # special tokens not in the vocab already
                @assert vidx == length(vocab_list) + 1
                push!(vocab_list, token)
            end
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
    !model_dict["fuse_unk"] && tokenizer_warn("fuse_unk is unsupported")
    byte_fallback = get(model_dict, "byte_fallback", false)
    unk_token = model_dict["unk_token"]
    sepsym = empty2nothing(model_dict["continuing_subword_prefix"])
    endsym = empty2nothing(model_dict["end_of_word_suffix"])
    merges = rank_from_lines(model_dict["merges"]; endsym)
    vocab_list = reverse_keymap_to_list(model_dict["vocab"])
    if byte_fallback
        bpe = ByteFallbackBPE(vocab_list, merges, sepsym, endsym)
    else
        cache = LRU{AbstractString, Vector{String}}(; maxsize = 1000)
        bpe = CachedBPE(BPE(merges, sepsym, endsym), cache)
    end
    return Base.Fix2(BPETokenization, bpe), bpe, unk_token, vocab_list
end

function extract_tokenization_method(::Val{:Unigram}, model_dict)
    unki = model_dict["unk_id"] + 1
    score_list = model_dict["vocab"]
    vocab_list = Vector{String}(undef, length(score_list))
    scores = Vector{Float64}(undef, length(score_list))
    for (i, entry) in enumerate(score_list)
        @assert length(entry) == 2
        vocab_list[i] = entry[1]
        scores[i] = entry[2]
    end
    unk = vocab_list[unki]
    unigram = Unigram(vocab_list, scores, unki)
    cache = LRU{AbstractString, Vector{String}}(; maxsize = 1000)
    unigram = CachedUnigram(unigram, cache)
    return Base.Fix2(UnigramTokenization, unigram), unigram, unk, vocab_list
end

# extract_tokenization_method(M::Val{:WordLevel}, model_dict)

function extract_tokenizer_model(model_dict)
    method, object, unk, vocab_list = extract_tokenization_method(model_dict)
    return method, object, unk, vocab_list
end

@valsplit extract_pre_tokenization(
    Val(tokenization_type::Symbol), pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict
) = load_error("Unsupported pre-tokenization method: $tokenization_type")

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
    @assert !pretokenizer_dict["add_prefix_space"] load_error_msg("add_prefix_space is unsupported")
    isnothing(tokenization) && (tokenization = GPT2Tokenization())
    normalizer = normalizer ∘ Base.Fix2(CodeNormalizer, gpt2_codemap())
    return tokenization, match_tokens, normalizer
end

function extract_pre_tokenization(
    ::Val{:Metaspace}, pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict
)
    @assert !haskey(pretokenizer_dict, "str_rep") || pretokenizer_dict["replacement"] == pretokenizer_dict["str_rep"]
    replacement = collect(pretokenizer_dict["replacement"])[]::Char
    add_prefix_space = pretokenizer_dict["add_prefix_space"]
    if isnothing(tokenization)
        tokenization = EachMatchTokenization(RuRegex("$replacement[^$replacement]*|[^$replacement]+"))
        normalizer = normalizer ∘ Base.Fix2(ReplaceNormalizer, ' '=>replacement)
        if add_prefix_space
            normalizer = normalizer ∘ Base.Fix2(SentenceFuncNormalizer, ensure_prefix(replacement))
        end
    else
        @assert tokenization == EachSplitTokenization(isspace) load_error_msg("Metaspace without WhiteSpaceSPlit is unsupported")
        metaspacef(x) = isspace(x) || x == replacement
        tokenization = EachSplitTokenization(metaspacef)
        if add_prefix_space
            normalizer = normalizer ∘ Base.Fix2(WordFuncNormalizer, ensure_prefix(replacement))
        end
    end
    return tokenization, match_tokens, normalizer
end

function extract_pre_tokenization(
    ::Val{:Split}, pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict
)
    @assert isnothing(tokenization) load_error_msg("Chaining tokenization is unsupported")
    behavior = pretokenizer_dict["behavior"]
    @assert behavior in ("Removed", "Isolated") load_error_msg("Only support removed or isolated behavior")
    @assert haskey(pretokenizer_dict["pattern"], "Regex") load_error_msg("Only support regex pattern")
    regex_str = pretokenizer_dict["pattern"]["Regex"]
    if behavior == "Removed"
        if pretokenizer_dict["invert"]
            if regex_str == raw"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
                tokenization = GPT2Tokenization()
            else
                if regex_str == raw"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"
                    regex = r"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"
                    if isnothing(match_tokens)
                        match_tokens = ["<|startoftext|>", "<|endoftext|>"]
                    else
                        push!(match_tokens, "<|startoftext|>", "<|endoftext|>")
                    end
                else
                    regex = RuRegex(regex_str)
                end
                tokenization = EachMatchTokenization(regex)
            end
        else
            tokenization = EachSplitTokenization(RuRegex(regex_str))
        end
    else
        tokenization = MatchSplitsTokenization(RuRegex(regex_str))
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
    pretokenizer_dict = tokenizer_dict["pre_tokenizer"]
    if !isnothing(pretokenizer_dict)
        pretokenization, match_tokens, normalizer = extract_pre_tokenization(
            pretokenizer_dict, nothing, match_tokens, identity, tokenizer_dict)
    else
        pretokenization = NoTokenization()
        normalizer = identity
    end
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
    @assert normalizer_dict["clean_text"] load_error_msg("bert normalize without clean_text")
    check = Ref{Bool}(false)
    scan(x -> (x isa BertUnCasedPreTokenization || x isa BertCasedPreTokenization) && (check[] = true),
         TokenizerStyle(), tokenization)
    check[] || load_error("BertNormalizer without BertPreTokenizer is unsupported")
    return tokenization
end

extract_normalizer(::Val{:Lowercase}, normalizer_dict, tokenization, tokenizer_dict) =
    TextEncodeBase.SentenceReplaceNormalizer(TextEncodeBase.LowercaseNormalizer(tokenization), "İ"=>"İ")

extract_normalizer(::Val{:NFD}, normalizer_dict, tokenization, tokenizer_dict) =
    TextEncodeBase.UnicodeNormalizer(tokenization, :NFD)

extract_normalizer(::Val{:NFKD}, normalizer_dict, tokenization, tokenizer_dict) =
    TextEncodeBase.UnicodeNormalizer(tokenization, :NFKD)

extract_normalizer(::Val{:NFC}, normalizer_dict, tokenization, tokenizer_dict) =
    TextEncodeBase.UnicodeNormalizer(tokenization, :NFC)

extract_normalizer(::Val{:NFKC}, normalizer_dict, tokenization, tokenizer_dict) =
    TextEncodeBase.UnicodeNormalizer(tokenization, :NFKC)

function extract_normalizer(::Val{:Replace}, normalizer_dict, tokenization, tokenizer_dict)
    @assert isone(length(normalizer_dict["pattern"])) load_error_msg("Multiple pattern")
    if haskey(normalizer_dict["pattern"], "Regex")
        pattern = RuRegex(normalizer_dict["pattern"]["Regex"])
    elseif haskey(normalizer_dict["pattern"], "String")
        pattern = normalizer_dict["pattern"]["String"]
    else
        load_error_msg("Only support regex or String pattern")
    end
    content = normalizer_dict["content"]
    return ReplaceNormalizer(tokenization, pattern=>content)
end

function extract_normalizer(::Val{:Precompiled}, normalizer_dict, tokenization, tokenizer_dict)
    precompiled = UnigramLanguageModel.PrecompiledNorm(normalizer_dict["precompiled_charsmap"])
    return PrecompiledNormalizer(tokenization, precompiled)
end

function extract_normalizer(::Val{:Prepend}, normalizer_dict, tokenization, tokenizer_dict)
    prepend = normalizer_dict["prepend"]
    return SentenceFuncNormalizer(tokenization, ensure_prefix(prepend))
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

extract_post_processor(::Nothing, tokenizer_dict, process_config) = process_config
extract_post_processor(post_processor_dict, tokenizer_dict, process_config) =
    extract_post_processor(Symbol(post_processor_dict["type"]), post_processor_dict, tokenizer_dict, process_config)
@valsplit extract_post_processor(
    Val(post_processor_type::Symbol), post_processor_dict, tokenizer_dict, process_config
) =
    load_error("Unsupported post processor method: $post_processor_type")

function extract_term(term)
    if haskey(term, "SpecialToken")
        @assert length(term) == 1
        term = term["SpecialToken"]
        id = term["id"]
        type_id = term["type_id"] + 1
        return ConstTerm(id, type_id)
    elseif haskey(term, "Sequence")
        @assert length(term) == 1
        term = term["Sequence"]
        id = term["id"]
        type_id = term["type_id"] + 1
        if id == "A"
            id = 1
        elseif id == "B"
            id = 2
        else
            load_error("Unknown pattern in TemplateProcessing: $term")
        end
        return IndexInputTerm{String}(id, type_id)
    else
        load_error("Unknown pattern in TemplateProcessing: $term")
    end
end

function extract_post_processor(::Val{:TemplateProcessing}, post_processor_dict, tokenizer_dict, process_config)
    all(Base.splat(==), zip(post_processor_dict["single"], post_processor_dict["pair"])) ||
        load_error("Un-mergeable pattern for TemplateProcessing")
    special_tokens = post_processor_dict["special_tokens"]
    single_term = map(extract_term, post_processor_dict["single"])
    pair_term = map(extract_term, post_processor_dict["pair"][length(single_term)+1:end])
    process = Pipelines(
        Pipeline{:token}(grouping_sentence, :token),
        Pipeline{(:token, :segment)}(SequenceTemplate(single_term..., RepeatedTerm(pair_term...)), :token),
    )
    process_config[:process] = process
    return process_config
end

function extract_post_processor(::Val{:BertProcessing}, post_processor_dict, tokenizer_dict, process_config)
    sepsym, sepid = post_processor_dict["sep"]
    startsym, startid = post_processor_dict["cls"]
    process = Pipelines(
        Pipeline{:token}(grouping_sentence, :token),
        Pipeline{(:token, :segment)}(
            SequenceTemplate(
                ConstTerm(startsym, 1), InputTerm{String}(1), ConstTerm(sepsym, 1),
                RepeatedTerm(InputTerm{String}(2), ConstTerm(sepsym, 2))),
            :token),
    )
    process_config[:process] = process
    return process_config
end

function extract_post_processor(::Val{:RobertaProcessing}, post_processor_dict, tokenizer_dict, process_config)
    @assert !post_processor_dict["add_prefix_space"] load_error_msg("add_prefix_space is unsupported")
    sepsym, sepid = post_processor_dict["sep"]
    startsym, startid = post_processor_dict["cls"]
    process = Pipelines(
        Pipeline{:token}(grouping_sentence, :token),
        Pipeline{(:token, :segment)}(
            SequenceTemplate(
                ConstTerm(startsym), InputTerm{String}(), ConstTerm(sepsym),
                RepeatedTerm(ConstTerm(sepsym), InputTerm{String}(), ConstTerm(sepsym))),
            :token),
    )
    process_config[:process] = process
    return process_config
end

function extract_post_processor(::Val{:ByteLevel}, post_processor_dict, tokenizer_dict, process_config)
    process = Pipelines(
        Pipeline{:token}(grouping_sentence, :token),
        Pipeline{:token}(SequenceTemplate(RepeatedTerm(InputTerm{String}()))(Val(1)), :token),
    )
    process_config[:process] = process
    return process_config
end

function extract_processor(tokenizer_json)
    process_config = extract_trunc_pad(tokenizer_json)
    process_config = extract_post_processor(tokenizer_json["post_processor"], tokenizer_json, process_config)
    return process_config
end

function reduce_nestedcall(fs)
    return foldl(fs; init = []) do init, f
        f isa typeof(identity) && return init
        isempty(init) && return push!(init, f)
        f0 = pop!(init)
        if f0 isa Base.Fix1{typeof(nestedcall)} && f isa Base.Fix1{typeof(nestedcall)}
            push!(init, nestedcall(f.x ∘ f0.x))
        else
            push!(init, f0, f)
        end
        return init
    end
end

function build_pipeline(fs)
    isempty(fs) && return identity
    length(fs) == 1 && return fs[]
    return foldl(Iterators.drop(fs, 1); init = Pipeline{:token}(first(fs), 1)) do pipe, f
        pipe |> Pipeline{:token}(f, :token)
    end |> PipeGet{:token}()
end

function extract_decoder(tokenizer_dict, config)
    decodes = Any[identity]
    textprocesses = Any[TextEncodeBase.join_text]
    decodes, textprocesses = extract_decoder(tokenizer_dict["decoder"], decodes, textprocesses)
    config[:clean_up_tokenization_spaces] && !(nestedcall(cleanup) in textprocesses) &&
        push!(textprocesses, nestedcall(cleanup))
    # https://github.com/huggingface/transformers/blob/25245ec26dc29bcf6102e1b4ddd0dfd02e720cf5/src/transformers/models/clip/tokenization_clip_fast.py#L95-L107
    if getconfigname(config) == :clip
        suffix = tokenizer_dict["model"]["end_of_word_suffix"]
        p = suffix => " "
        remove_suffix(s) = replace(s, p)
        remove_tail_space(s) = string_strip(' ', s; start=0, stop=1)
        push!(decodes, nestedcall(remove_suffix))
        push!(textprocesses, nestedcall(remove_tail_space))
    end
    decode = build_pipeline(reduce_nestedcall(decodes))
    textprocess = build_pipeline(reduce_nestedcall(textprocesses))
    return decode, textprocess
end

extract_decoder(::Nothing, decode, textprocess) = decode, textprocess
extract_decoder(decoder_dict, decode, textprocess) = extract_decoder(Symbol(decoder_dict["type"]), decoder_dict, decode, textprocess)

@valsplit extract_decoder(Val(decoder_type::Symbol), decoder_dict, decode, textprocess) = load_error("Unsupported decoder method: $decoder_type")

function extract_decoder(::Val{:Replace}, decoder_dict, decode, textprocess)
    @assert isone(length(decoder_dict["pattern"])) load_error_msg("Multiple pattern")
    if haskey(decoder_dict["pattern"], "Regex")
        pattern = RuRegex(normalizer_dict["pattern"]["Regex"])
    elseif haskey(decoder_dict["pattern"], "String")
        pattern = decoder_dict["pattern"]["String"]
    else
        load_error_msg("Only support regex or String pattern")
    end
    content = decoder_dict["content"]
    p = pattern => content
    Replace(s) = replace(s, p)
    push!(decode, nestedcall(Replace))
    return decode, textprocess
end

function extract_decoder(::Val{:ByteFallback}, decoder_dict, decode, textprocess)
    push!(decode, nestedcall(fallback2byte))
    return decode, textprocess
end

function extract_decoder(::Val{:Strip}, decoder_dict, decode, textprocess)
    content = decoder_dict["content"]
    @assert length(content) == 1 load_error_msg("Strip decoder with string content")
    char = content[1]
    start = decoder_dict["start"]
    stop = decoder_dict["stop"]
    Strip(s) = string_strip(char, s; start, stop)
    push!(textprocess, nestedcall(Strip))
    return decode, textprocess
end

function extract_decoder(::Val{:Fuse}, decoder_dict, decode, textprocess)
    !(TextEncodeBase.join_text in textprocess) && push!(textprocess, TextEncodeBase.join_text)
    return decode, textprocess
end

function extract_decoder(::Val{:Metaspace}, decoder_dict, decode, textprocess)
    @assert !haskey(decoder_dict, "str_rep") || decoder_dict["replacement"] == decoder_dict["str_rep"]
    replacement = collect(decoder_dict["replacement"])[]::Char
    p = replacement => ' '
    metaspace2space(s) = replace(s, p)
    push!(decode, nestedcall(metaspace2space))
    if decoder_dict["add_prefix_space"]
        remove_prefix_space(s) = string_strip(' ', s; start=1, stop=0)
        push!(textprocess, nestedcall(remove_prefix_space))
    end
    return decode, textprocess
end

function extract_decoder(::Val{:BPEDecoder}, decoder_dict, decode, textprocess)
    suffix = decoder_dict["suffix"]
    p = suffix => " "
    remove_suffix(s) = replace(s, p)
    remove_tail_space(s) = string_strip(' ', s; start=0, stop=1)
    push!(decode, nestedcall(remove_suffix))
    push!(textprocess, nestedcall(remove_tail_space))
    return decode, textprocess
end

function extract_decoder(::Val{:ByteLevel}, decoder_dict, decode, textprocess)
    push!(decode, nestedcall(TextEncodeBase.CodeUnMap(gpt2_codemap())))
    return decode, textprocess
end

function extract_decoder(::Val{:WordPiece}, decoder_dict, decode, textprocess)
    prefix = decoder_dict["prefix"]
    function remove_conti_prefix(s)
        if startswith(s, prefix)
            return String(SubString(s, 1 + ncodeunits(prefix)))
        else
            return " $s"
        end
    end
    push!(decode, nestedcall(remove_conti_prefix))
    remove_prefix_space(s) = string_strip(' ', s; start=1, stop=0)
    push!(textprocess, nestedcall(remove_prefix_space))
    return decode, textprocess
end

function extract_decoder(::Val{:Sequence}, decoder_dict, decode, textprocess)
    for sub_decoder_dict in decoder_dict["decoders"]
        decode, textprocess = extract_decoder(sub_decoder_dict, decode, textprocess)
    end
    return decode, textprocess
end

function load_fast_tokenizer_components(tokenizer_json, config)
    tokenizer_dict = json_load(tokenizer_json)
    method, tokenization_object, unk, vocab_list = extract_tokenizer_model(tokenizer_dict["model"])
    match_tokens = extract_and_add_tokens!(tokenizer_dict["added_tokens"], vocab_list)
    base_tokenization, match_tokens = extract_base_tokenization(method, match_tokens, tokenizer_dict)
    match_tokens = empty_then_nothing(match_tokens)
    process_config = extract_processor(tokenizer_dict)
    decode, textprocess = extract_decoder(tokenizer_dict, config)
    return base_tokenization, match_tokens, vocab_list, unk, tokenization_object, process_config, decode, textprocess
end

load_fast_tokenizer(type, tokenizer_json, config) = load_fast_tokenizer(tokenizer_json, config) # default ignoring type
function load_fast_tokenizer(tokenizer_json, config)
    base_tokenization, match_tokens, vocab_list, unk, tokenization_object, process_config, decode, textprocess =
        load_fast_tokenizer_components(tokenizer_json, config)
    isnothing(match_tokens) || (base_tokenization = MatchTokenization(base_tokenization, match_tokens))
    isnothing(unk) && (unk = "<unk>") # dummy unk token, wouldn't appear in vocabulary
    unk isa AbstractString || (unk = vocab_list[unk])
    vocab = Vocab(vocab_list, unk)
    tokenizer = TextTokenizer(base_tokenization)
    return tokenizer, vocab, process_config, decode, textprocess
end
