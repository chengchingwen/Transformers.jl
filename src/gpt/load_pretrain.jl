using WordTokenizers

isbson(s) = endswith(s, ".bson")

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

function load_gpt_pretrain(path::AbstractString)
  if isnpbson(path)
    @info "loading raw gpt np data: $path"
    bson = BSON.load(path)
    bson[:weights], bson[:bpe], bson[:raw_vocab], bson[:tokenizer]
  elseif isbson(path)
    @info "loading gpt model: $path"
    bson = BSON.load(path)
    bson[:gpt_model], bson[:bpe], bson[:vocab], bson[:tokenizer]
  else
    error("""
          Unknown data type: Is it a gpt model?
          If it is the weight file get from finetune-transformer-lm, then run npy2bson on that file beforehand.
    """)
  end
end
