using BytePairEncoding

function load_special_tokens_map(special_tokens_map_json)
    special_tokens = Dict{Symbol, String}()
    for (k, v) in json_load(special_tokens_map_json)
        special_tokens[k] = v isa String ? v : v["content"]
    end
    return special_tokens
end

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
