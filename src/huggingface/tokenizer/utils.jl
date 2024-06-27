using BytePairEncoding
using TextEncodeBase
using StructWalk

import ..TextEncoders

function load_special_tokens_map(special_tokens_map_json)
    special_tokens = Dict{Symbol, String}()
    for (k, v) in JSON3.read(read(special_tokens_map_json))
        val = v isa AbstractDict ? v["content"] : v
        if val isa String
            special_tokens[k] = val
        end
    end
    return special_tokens
end

function reverse_keymap_to_list(dict)
    vocab_list = Vector{String}(undef, length(dict))
    for (k, v) in dict
        v += 1
        @assert !isassigned(vocab_list, v) "Two word has same index: $(k) and $(vocab_list[v])"
        vocab_list[v] = String(k)
    end
    @assert all(Base.Fix1(isassigned, vocab_list), eachindex(vocab_list)) "There is a gap in the vocabulary"
    return vocab_list
end

function rank_from_lines(lines; endsym = nothing)
    rank = Dict{NTuple{2, BytePairEncoding.Merge}, Int}()
    pattern = isnothing(endsym) ? nothing : Base.compile(Regex("(.*)\\Q$endsym\\E\$"))
    for (i, line) in enumerate(lines)
        p = BytePairEncoding.parse_merge(line, pattern)
        rank[p] = i
    end
    return rank
end

empty_then_nothing(::Nothing) = nothing
empty_then_nothing(x) = isempty(unique!(x)) ? nothing : x

tokenizer_warn(msg) = @warn "$msg, the tokenization result might be slightly different in some cases."

function get_tkr_token(tkr_cfg, special_tokens, name, default)
    token = nothing
    if !isnothing(special_tokens)
        token = get(special_tokens, name, nothing)
    end
    if isnothing(token)
        token = isnothing(tkr_cfg) ? default : get(tkr_cfg, name, default)
    end
    return token
end

function heuristic_extract_fast_tkr_kwargs(config, tkr_cfg, special_tokens)
    kwargs = Dict{Symbol, Any}()
    kwargs[:startsym] = get_tkr_token(tkr_cfg, special_tokens, :bos_token, "<s>")
    kwargs[:endsym] = get_tkr_token(tkr_cfg, special_tokens, :eos_token, "</s>")
    kwargs[:padsym] = get_tkr_token(tkr_cfg, special_tokens, :pad_token, "<pad>")
    return kwargs
end

function heuristic_extract_slow_tkr_kwargs(config, tkr_cfg, special_tokens)
    slow_tkr_kwargs = Dict{Symbol, Any}()
    slow_tkr_kwargs[:unk_token] = get_tkr_token(tkr_cfg, special_tokens, :unk_token, "<unk>")
    return slow_tkr_kwargs
end

function guess_encoder_construct(tkr)
    ref = Ref{Symbol}(:default)
    StructWalk.scan(TextEncodeBase.TokenizerStyle(), tkr) do x
        if x isa TextEncoders.WordPieceTokenization
            ref[] = :bert
        elseif x isa TextEncoders.GPT2Tokenization
            ref[] = :gpt2
        elseif x isa TextEncoders.UnigramTokenization
            ref[] = :t5
        end
    end
    Tsym = ref[]
    if Tsym == :bert
        return TextEncoders.BertTextEncoder
    elseif Tsym == :gpt2
        return TextEncoders.GPT2TextEncoder
    elseif Tsym == :t5
        return TextEncoders.T5TextEncoder
    else
        return TextEncoders.TransformerTextEncoder
    end
end

function _check_padsym(kwargs)
    kw = values(kwargs)
    # make sure padsym is not nothing
    if haskey(kw, :padsym) && isnothing(kw.padsym)
        @warn "padsym is set to `nothing`, using \"<pad>\" instead"
        kw = merge(kw, (padsym = "<pad>",))
    end
    return kw
end

function _hgf_preprocess(
    ; padsym, trunc = nothing, fixedsize = false, trunc_end = :tail, pad_end = :tail,
    process = nothing, kws...
)
    truncf = TextEncoders.get_trunc_pad_func(fixedsize, trunc, trunc_end, pad_end)
    maskf = TextEncoders.get_mask_func(trunc, pad_end)
    has_segment = false
    if !isnothing(process)
        process = Pipeline{:token}(nestedcall(TextEncoders.string_getvalue), 1) |> process
        if :segment in FuncPipelines.target_name.(process.pipes)
            has_segment = true
            process = process |>
                Pipeline{:segment}(truncf(1), :segment) |>
                Pipeline{:segment}(nested2batch, :segment)
        end
    else
        process = Pipeline{:token}(nestedcall(TextEncoders.string_getvalue), 1)
    end
    return process |>
        Pipeline{:attention_mask}(maskf, :token) |>
        Pipeline{:token}(truncf(padsym), :token) |>
        Pipeline{:token}(nested2batch, :token) |>
        Pipeline{:sequence_mask}(identity, :attention_mask) |>
        (has_segment ? PipeGet{(:token, :segment, :attention_mask, :sequence_mask)}() : PipeGet{(:token, :attention_mask, :sequence_mask)}())
end

function heuristic_encoder_construct(tokenizer, vocab, kwargs)
    constr = guess_encoder_construct(tokenizer)
    kwargs = _check_padsym(kwargs)
    process = _hgf_preprocess(; kwargs...)
    nkws = Base.structdiff(kwargs, NamedTuple{(:process, :fixedsize, :trunc_end, :pad_end)})
    return constr(tokenizer, vocab, process; nkws...)
end
