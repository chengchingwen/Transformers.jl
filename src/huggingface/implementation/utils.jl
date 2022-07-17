using HuggingFaceApi: list_model_files

ensure_possible_files(possible_files, model_name; revision = nothing, auth_token = nothing, kw...) =
    _ensure(list_model_files, possible_files, model_name; revision, token = auth_token)

ensure_config(config, model_name; kw...) = _ensure(load_config, config, model_name; kw...)

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

load_tokenizer_added_tokens(model_name; kw...) = JSON.parsefile(hgf_tokenizer_added_token(model_name; kw...); dicttype = Dict{String, Any})
