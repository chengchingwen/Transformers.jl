using ..Transformers
using ..Transformers.Basic
using TextEncodeBase
using JSON

using HuggingFaceApi

load_tokenizer_config(model_name; kw...) = json_load(hgf_tokenizer_config(model_name; kw...))

function load_tokenizer(model_name; possible_files = nothing, config = nothing, kw...)
    possible_files = ensure_possible_files(possible_files, model_name; kw...)

    if TOKENIZER_CONFIG_FILE in possible_files
        tkr_cfg = load_tokenizer_config(model_name; kw...)
        tkr_type = get(tkr_cfg, :tokenizer_class, nothing)
    else
        tkr_cfg = nothing
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

    return load_tokenizer(tkr_type, model_name; possible_files, config, tkr_cfg, kw...)
end

load_tokenizer(tkr_type::Symbol, model_name; kw...) = load_tokenizer(tokenizer_type(tkr_type), model_name; kw...)

@valsplit tokenizer_type(Val(type::Symbol)) = error("No tokenizer found for $type")
