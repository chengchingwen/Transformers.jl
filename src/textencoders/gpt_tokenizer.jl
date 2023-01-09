using WordTokenizers

"""
The function in the origin gpt code

fixes some issues the spacy tokenizer had on books corpus
also does some whitespace standardization
"""
function text_standardize(text)
    text = lowercase(text)
    text = replace(text, r"([a-z])(,|\.)"=>s"\1 \2")
    text = replace(text, "—"=>"-")
    text = replace(text, "–"=>"-")
    text = replace(text, "―"=>"-")
    text = replace(text, "…"=>"...")
    text = replace(text, "´"=>"'")
    text = replace(text, r"""(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)"""=>s" \1 ")
    text = replace(text, r"\s*\n\s*"=>" \n ")
    text = replace(text, r"[^\S\n]+"=>" ")
    strip(text)
end

"""
    gpt_tokenizer(x)

An alternative for origin tokenizer (spacy tokenizer) used in gpt model.
"""
gpt_tokenizer(x) = toktok_tokenize(text_standardize(x))
