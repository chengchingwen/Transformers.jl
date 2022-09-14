using ..Transformers
using TextEncodeBase
using JSON

using HuggingFaceApi

load_tokenizer_config(model_name; kw...) = json_load(hgf_tokenizer_config(model_name; kw...))

function load_tokenizer(model_name; possible_files = nothing, config = nothing, kw...)
    possible_files = ensure_possible_files(possible_files, model_name; kw...)

    if TOKENIZER_CONFIG_FILE in possible_files
        tkr_config = load_tokenizer_config(model_name; kw...)
        tkr_type = get(tkr_config, :tokenizer_class, nothing)
    else
        tkr_config = nothing
        tkr_type = nothing
    end

    if isnothing(tkr_type)
        config = ensure_config(config, model_name; kw...)
        tkr_type = something(config.tokenizer_class, Symbol(config.model_type))
    end

    if tkr_type isa AbstractString
        m = match(r"(\S+)Tokenizer(Fast)?", tkr_type)
        isnothing(m) && error("Unknown tokenizer: $tkr_type")
        tkr_type = Symbol(lowercase(m.captures[1]))
    end

    return load_tokenizer(tkr_type, model_name; possible_files, config, tkr_config, kw...)
end

@valsplit tokenizer_type(Val(type::Symbol)) = error("No tokenizer found for $type")

load_tokenizer(tkr_type::Symbol, model_name; kw...) = load_tokenizer(tokenizer_type(tkr_type), model_name; kw...)

function load_tokenizer(
    T::Val, model_name; force_fast_tkr = false, possible_files = nothing,
    config = nothing, tkr_config = nothing,
    kw...
)
    possible_files = ensure_possible_files(possible_files, model_name; kw...)
    config = ensure_config(config, model_name; kw...)

    isnothing(tkr_config) && TOKENIZER_CONFIG_FILE in possible_files &&
        (tkr_config = load_tokenizer_config(model_name; kw...))
    special_tokens = SPECIAL_TOKENS_MAP_FILE in possible_files ?
        load_special_tokens_map(hgf_tokenizer_special_tokens_map(model_name; kw...)) : nothing
    kwargs, slow_tkr_kwargs = extract_tkr_kwargs(T, tkr_config, config, special_tokens)

    if FULL_TOKENIZER_FILE in possible_files || force_fast_tkr
        @assert FULL_TOKENIZER_FILE in possible_files "Forcely using fast tokenizer but cannot find $FULL_TOKENIZER_FILE in $model_name repo"
        tokenizer, vocab, process_config = load_fast_tokenizer(T, hgf_tokenizer(model_name; kw...))
    else
        slow_files = slow_tkr_files(T)
        @assert all(Base.Fix2(in, possible_files), slow_files) "Cannot not find $slow_files or $FULL_TOKENIZER_FILE in $model_name repo"
        slow_files = map(file->hgf_file(model_name, file; kw...), slow_files)
        added_tokens_file = ADDED_TOKENS_FILE in possible_files ?
            hgf_tokenizer_added_token(model_name; kw...) : nothing
        tokenizer, vocab, process_config = load_slow_tokenizer(
            T, slow_files..., added_tokens_file, special_tokens; slow_tkr_kwargs...)
    end

    for (k, v) in process_config
        kwargs[k] = v
    end

    return encoder_construct(T, tokenizer, vocab; kwargs...)
end
