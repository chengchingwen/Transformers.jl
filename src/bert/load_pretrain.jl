isbson(s) = endswith(s, ".bson")
istfbson(s) = endswith(s, ".tfbson")

function load_bert_pretrain(path::AbstractString)
  if istfbson(path)
    @info "loading raw bert data: $path"
    bson = BSON.load(path)
    bson[:config], bson[:weights], bson[:vocab]
  elseif isbson(path)
    @info "loading bert model"
    bson = BSON.load(path)
    bson[:bert_model], bson[:wordpiece], bson[:tokenizer]
  else
    error("""
          Unknown data type: Is it a bert model?
          If it is the weight file get from google-bert, then run tfckpt2bson on that file beforehand.
    """)
  end
end
