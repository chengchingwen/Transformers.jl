using BytePairEncoding
using BytePairEncoding: CachedBPE

function load_bpe_from_old_ver_file(bson)
    _rank = BSON.raise_recursive(bson[:bpe][:data][6], Main)
    rank = Dict{NTuple{2, BytePairEncoding.Merge}, Int}()
    endsym = BSON.raise_recursive(bson[:bpe][:data][2], Main)
    new_endsym = BSON.raise_recursive(bson[:bpe][:data][3], Main)
    sepsym = BSON.raise_recursive(bson[:bpe][:data][1], Main)
    pattern = isnothing(endsym) ? nothing : Base.compile(Regex("(.*)\\Q$endsym\\E\$"))
    for (k, v) in _rank
        p = BytePairEncoding.parse_merge(k, pattern)
        rank[p] = v
    end
    bpe = CachedBPE(BPE(rank, sepsym, new_endsym))
    return bpe
end

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
      bpe = load_bpe_from_old_ver_file(bson)
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
      bpe = load_bpe_from_old_ver_file(bson)
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
