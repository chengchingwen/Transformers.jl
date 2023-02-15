function _load_added_tokens(added_tokens_json, vocab_list)
    added_tokens = String[]
    added_tokens_dict = JSON3.read(read(added_tokens_json))
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
    match_tokens = isnothing(added_tokens_file) ? nothing : _load_added_tokens(added_tokens_file, vocab_list)
    if !isnothing(special_tokens)
        special_token_values = values(special_tokens)
        match_tokens = isnothing(match_tokens) ?
            collect(special_token_values) : append!(match_tokens, special_token_values)
    end
    match_tokens = empty_then_nothing(match_tokens)
    return match_tokens
end

@valsplit load_slow_tokenizer(Val(type::Symbol), args...; kwargs...) = error("No slow tokenizer defined for $type.")
