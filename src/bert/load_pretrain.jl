"""
    load_bert_pretrain(path, sym = :all)

Loading bert data/model from pretrain bson data. use `sym` to determine which data to load.
possible value: `:all`(default), `bert_model`, `wordpiece`, `tokenizer`.
"""
function load_bert_pretrain(path::AbstractString, sym = :all)
  @info "loading pretrain bert model: $(basename(path)) $(sym == :all ? "" : sym)"
  @assert sym ∈ (:all, :bert_model, :wordpiece, :tokenizer) "sym only support :all, :bert_model, :wordpiece, :tokenizer"
  if istfbson(path)

    bson = BSON.parse(path)

    config = BSON.raise_recursive(bson[:config])
    tokenizer = named2tokenizer(config["filename"])

    sym == :tokenizer && return tokenizer

    wordpiece = WordPiece(convert(Vector{String}, BSON.raise_recursive(bson[:vocab])))

    sym == :wordpiece && return wordpiece

    weights = BSON.raise_recursive(bson[:weights])
    bert_model = load_bert_from_tfbson(config, weights)

    sym == :bert_model && return bert_model

    return bert_model, wordpiece, tokenizer
  elseif isbson(path)
    bson = BSON.parse(path)
    tokenizer = BSON.raise_recursive(bson[:tokenizer])
    sym == :tokenizer && return tokenizer
    wordpiece = BSON.raise_recursive(bson[:wordpiece])
    sym == :wordpiece && return wordpiece
    bert_model = BSON.raise_recursive(bson[:bert_model])
    sym == :bert_model && return bert_model
    return bert_model, wordpiece, tokenizer
  else
    error("""
          Unknown data type: Is it a bert model?
          If it is the weight file get from google-bert, then run `tfckpt2bson` on that file beforehand.
    """)
  end
end

# function load_pretrain_file(path::AbstractString)
#   if istfbson(path)
#     @info "loading raw data: $path"
#     bson = BSON.load(path)
#     bson[:config], bson[:weights], bson[:vocab]
#   elseif isbson(path)
#     @info "loading model"
#     bson = BSON.load(path)
#     bson[:bert_model], bson[:wordpiece], bson[:tokenizer]
#   elseif iszip(path)
#     data = ZipFile.Reader(path)
#     config, weights, vocab = readckptfolder(data)
#   elseif isdir(path)
#     config, weights, vocab = readckptfolder(path)
#   else
#     error("""Unknown file format""")
#   end
# end
