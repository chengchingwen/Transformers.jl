using ..Transformers
using ..TextEncoders
using FuncPipelines
using TextEncodeBase
using TextEncodeBase: trunc_and_pad, trunc_or_pad, nested2batch, nestedcall
using ValSplit
using BangBang

load_tokenizer_config(model_name; kw...) = JSON3.read(read(hgf_tokenizer_config(model_name; kw...)))

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

function load_tokenizer(
    tkr_type, model_name; force_fast_tkr = false, possible_files = nothing,
    config = nothing, tkr_config = nothing,
    kw...
)
    T = tokenizer_type(tkr_type)
    possible_files = ensure_possible_files(possible_files, model_name; kw...)
    config = ensure_config(config, model_name; kw...)

    isnothing(tkr_config) && TOKENIZER_CONFIG_FILE in possible_files &&
        (tkr_config = load_tokenizer_config(model_name; kw...))
    special_tokens = SPECIAL_TOKENS_MAP_FILE in possible_files ?
        load_special_tokens_map(hgf_tokenizer_special_tokens_map(model_name; kw...)) : nothing
    tkr_config = isnothing(tkr_config) ? (;) : tkr_config
    kwargs = extract_fast_tkr_kwargs(T, tkr_config, config, special_tokens)

    if FULL_TOKENIZER_FILE in possible_files || force_fast_tkr
        @assert FULL_TOKENIZER_FILE in possible_files "Forcely using fast tokenizer but cannot find $FULL_TOKENIZER_FILE in $model_name repo"
        tokenizer, vocab, process_config, decode, textprocess =
            load_fast_tokenizer(T, hgf_tokenizer(model_name; kw...), config)
    else
        slow_tkr_kwargs = extract_slow_tkr_kwargs(T, tkr_config, config, special_tokens)
        slow_files = slow_tkr_files(T)
        @assert all(Base.Fix2(in, possible_files), slow_files) "Cannot not find $slow_files or $FULL_TOKENIZER_FILE in $model_name repo"
        slow_files = map(file->hgf_file(model_name, file; kw...), slow_files)
        added_tokens_file = ADDED_TOKENS_FILE in possible_files ?
            hgf_tokenizer_added_token(model_name; kw...) : nothing
        decode = identity
        textprocess = TextEncodeBase.join_text
        tokenizer, vocab, process_config = load_slow_tokenizer(
            T, slow_files..., added_tokens_file, special_tokens; slow_tkr_kwargs...)
    end

    for (k, v) in process_config
        kwargs[k] = v
    end

    tkr = encoder_construct(T, tokenizer, vocab; kwargs...)
    return setproperties!!(tkr, (; decode, textprocess))
end

tokenizer_type(type::Val) = type
@valsplit tokenizer_type(Val(type::Symbol)) = type

extract_fast_tkr_kwargs(type, tkr_cfg, config, special_tokens) =
    extract_fast_tkr_kwargs(type, config, special_tokens; tkr_cfg...)
extract_fast_tkr_kwargs(_type::Val{type}, config, special_tokens; tkr_cfg...) where type =
    extract_fast_tkr_kwargs(type, config, special_tokens; tkr_cfg...)
function extract_fast_tkr_kwargs(type::Symbol, config, special_tokens; tkr_cfg...)
    @debug "No extract_fast_tkr_kwargs handler registed for $type, using heuristic"
    vals = valarg_params(extract_fast_tkr_kwargs, Tuple{Val, Any, Any}, 1, Symbol)
    default_f = () -> heuristic_extract_fast_tkr_kwargs(config, tkr_cfg, special_tokens)
    return ValSplit._valswitch(Val(vals), Val(3), Core.kwfunc(extract_fast_tkr_kwargs), default_f,
                               tkr_cfg, extract_fast_tkr_kwargs, type, config, special_tokens)
end

extract_slow_tkr_kwargs(type, tkr_cfg, config, special_tokens) =
    extract_slow_tkr_kwargs(type, config, special_tokens; tkr_cfg...)
extract_slow_tkr_kwargs(_type::Val{type}, config, special_tokens; tkr_cfg...) where type =
    extract_slow_tkr_kwargs(type, config, special_tokens; tkr_cfg...)
function extract_slow_tkr_kwargs(type::Symbol, config, special_tokens; tkr_cfg...)
    @debug "No extract_slow_tkr_kwargs handler registed for $type, using heuristic"
    vals = valarg_params(extract_slow_tkr_kwargs, Tuple{Val, Any, Any}, 1, Symbol)
    default_f = () -> heuristic_extract_slow_tkr_kwargs(config, tkr_cfg, special_tokens)
    return ValSplit._valswitch(Val(vals), Val(3), Core.kwfunc(extract_slow_tkr_kwargs), default_f,
                               tkr_cfg, extract_slow_tkr_kwargs, type, config, special_tokens)
end

@valsplit slow_tkr_files(Val(type::Symbol)) = error("Don't know what files are need to load slow $type tokenizer.")

encoder_construct(_type::Val{type}, tokenizer, vocab; kwargs...) where type =
    encoder_construct(type, tokenizer, vocab; kwargs...)
function encoder_construct(type::Symbol, tokenizer, vocab; kwargs...)
    @debug "No encoder_construct handdler registed for $type, using default"
    vals = valarg_params(encoder_construct, Tuple{Val, Any, Any}, 1, Symbol)
    default_f = () -> heuristic_encoder_construct(tokenizer, vocab, kwargs)
    return ValSplit._valswitch(Val(vals), Val(3), Core.kwfunc(encoder_construct), default_f,
                               kwargs, encoder_construct, type, tokenizer, vocab)
end

include("utils.jl")
include("slow_tkr.jl")
include("fast_tkr.jl")

# api doc

"""
    load_tokenizer(model_name; config = nothing, local_files_only = false, cache = true)

Load the text encoder of `model_name` from huggingface hub. By default, this function would check if `model_name`
 exists on huggingface hub, download all required files for this text encoder (and cache these files if `cache` is
 set), and then load and return the text encoder. If `local_files_only = false`, it would check whether all cached
 files are up-to-date and update if not (and thus require network access every time it is called). By setting
 `local_files_only = true`, it would try to find the files from the cache directly and error out if not found.
 For managing the caches, see the `HuggingFaceApi.jl` package.

# Example

```julia-repl
julia> load_tokenizer("t5-small")
T5TextEncoder(
├─ TextTokenizer(MatchTokenization(PrecompiledNormalizer(WordReplaceNormalizer(UnigramTokenization(EachSplitTokenization(splitter = isspace), unigram = Unigram(vocab_size = 32100, unk = <unk>)), pattern = r"^(?!▁)(.*)\$" => s"▁\1"), precompiled = PrecompiledNorm(...)), 103 patterns)),
├─ vocab = Vocab{String, SizedArray}(size = 32100, unk = <unk>, unki = 3),
├─ endsym = </s>,
├─ padsym = <pad>,
└─ process = Pipelines:
  ╰─ target[token] := TextEncodeBase.nestedcall(string_getvalue, source)
  ╰─ target[token] := Transformers.TextEncoders.grouping_sentence(target.token)
  ╰─ target[(token, segment)] := SequenceTemplate{String}(Input[1]:<type=1> </s>:<type=1> (Input[2]:<type=1> </s>:<type=1>)...)(target.token)
  ╰─ target[attention_mask] := (NeuralAttentionlib.LengthMask ∘ Transformers.TextEncoders.getlengths(nothing))(target.token)
  ╰─ target[token] := TextEncodeBase.trunc_and_pad(nothing, <pad>, tail, tail)(target.token)
  ╰─ target[token] := TextEncodeBase.nested2batch(target.token)
  ╰─ target := (target.token, target.attention_mask)
)

```
"""
load_tokenizer
