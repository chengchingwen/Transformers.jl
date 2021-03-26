"""
    load_gpt_pretrain(path::AbstractString, sym = :all;
                      startsym="_start_",
                      delisym="_delimiter_",
                      clfsym="_classify_",
                      unksym="<unk>")

load gpt data/model from pretrain bson data. use `sym` to determine which data to load. 
set `<xxx>sym`s for setting special symbols in vocabulary.

possible value: `:all`(default), `gpt_model`, `bpe`, `vocab`, `tokenizer`.
"""
function load_gpt_pretrain(path::AbstractString, sym = :all; kw...)
  @info "loading pretrain gpt model: $(basename(path)) $(sym == :all ? "" : sym)"
  @assert sym ∈ (:all, :gpt_model, :bpe, :vocab, :tokenizer) "sym only support :all, :gpt_model, :bpe, :vocab, :tokenizer"
  if isnpbson(path)

    bson = BSON.parse(path)

    if sym ∈ (:all, :bpe)
      bpe = BSON.raise_recursive(bson[:bpe], Main)
      sym == :bpe && return bpe
    end

    tokenizer = BSON.raise_recursive(bson[:tokenizer], Main)
    sym == :tokenizer && return tokenizer

    raw_vocab = BSON.raise_recursive(bson[:raw_vocab], Main)
    vocab = build_vocab(raw_vocab; kw...)
    sym == :vocab && return vocab

    weights = BSON.raise_recursive(bson[:weights], Main)
    gpt_model = load_gpt_from_npbson(weights, length(vocab))
    sym == :gpt_model && return gpt_model

    return gpt_model, bpe, vocab, tokenizer
  elseif isbson(path)
    bson = BSON.parse(path)
    if sym ∈ (:all, :bpe)
      bpe = BSON.raise_recursive(bson[:bpe], Main)
      sym == :bpe && return bpe
    end
    tokenizer = BSON.raise_recursive(bson[:tokenizer], Main)
    sym == :tokenizer && return tokenizer
    vocab = BSON.raise_recursive(bson[:vocab], Main)
    sym == :vocab && return vocab
    gpt_model = BSON.raise_recursive(bson[:gpt_model], Main)
    sym == :gpt_model && return gpt_model
    return gpt_model, bpe, vocab, tokenizer
  else
    error("""
          Unknown data type: Is it a gpt model?
          If it is the weight file get from finetune-transformer-lm, then run `npy2bson` on that file beforehand.
    """)
  end
end

# function load_gpt_pretrain(path::AbstractString)
#   if isnpbson(path)
#     @info "loading raw gpt np data: $path"
#     bson = BSON.load(path)
#     bson[:weights], bson[:bpe], bson[:raw_vocab], bson[:tokenizer]
#   elseif isbson(path)
#     @info "loading gpt model: $path"
#     bson = BSON.load(path)
#     bson[:gpt_model], bson[:bpe], bson[:vocab], bson[:tokenizer]
#   else
#     error("""
#           Unknown data type: Is it a gpt model?
#           If it is the weight file get from finetune-transformer-lm, then run npy2bson on that file beforehand.
#     """)
#   end
# end
