# turn a tf bert format to bson

using JSON
using BSON
# using ZipFile
using TensorFlow

using Flux
using Flux: loadparams!

using ..Basic

iszip(s) = endswith(s, ".zip")

function tfckpt2bson(path; saveto="./", confname = "bert_config.json", ckptname = "bert_model.ckpt", vocabname = "vocab.txt")
  if iszip(path)
    error("not implement yet")
  else
    ckptfolder(path, saveto)
  end
end

#should be changed to use c api once the patch is included
function readckpt(path)
  weights = Dict{String, Array}()
  TensorFlow.init()
  ckpt = TensorFlow.pywrap_tensorflow.x.NewCheckpointReader(path)
  shapes = ckpt.get_variable_to_shape_map()
  #dtype = ckpt.get_variable_to_dtype_map()

  for (name, shape) ∈ shapes
    weight = ckpt.get_tensor(name)
    if length(shape) == 2
      weight = collect(weight')
    end
    weights[name] = weight
  end

  weights
end

function ckptfolder(dir, saveto; confname = "bert_config.json", ckptname = "bert_model.ckpt", vocabname = "vocab.txt")
  files = readdir(dir)

  confname ∉ files && error("config file $confname not found")
  ckptname*".meta" ∉ files && error("ckpt file $ckptname not found")
  vocabname ∉ files && error("vocab file $vocabname not found")

  filename = basename(isdirpath(dir) ? dir[1:end-1] : dir)
  bsonname = normpath(joinpath(saveto, filename * ".bson"))

  config = JSON.parsefile(joinpath(dir, confname))
  config["filename"] = filename
  weights = readckpt(joinpath(dir, ckptname))
  vocab = readlines(open(joinpath(dir, vocabname)))
  BSON.@save bsonname config weights vocab
  bsonname
end


bson2bert(path::AbstractString) = bson2bert(BSON.load(path))
function bson2bert(bson::Dict{Symbol, Any})
    vocab = load_vocab(bson)
    bert = load_model(bson)
    bert, vocab
end

function load_vocab(bson) end

function get_activation(act_string)
    if act_string == "gelu"
        gelu
    elseif act_string == "relu"
        relu
    elseif act_string == "tanh"
        tanh
    elseif act_string == "linear"
        identity
    else
        throw(DomainError(act_string, "activation support: linear, gelu, relu, tanh"))
    end
end

function load_model(bson)
    config = bson[:config]

    bert = Bert(
        config["hidden_size"],
        config["num_attention_heads"],
        config["intermediate_size"],
        config["num_hidden_layers"];
        act = get_activation(config["hidden_act"]),
        pdrop = config["hidden_dropout_prob"],
        att_pdrop = config["attention_probs_dropout_prob"]
    )

    tok_emb = Embed(
        config["hidden_size"],
        config["vocab_size"]
    )

    seg_emb = Embed(
        config["hidden_size"],
        config["type_vocab_size"]
    )

    posi_emb = PositionEmbedding(
        config["hidden_size"],
        config["max_position_embeddings"];
        trainable = true
    )

    emb_post = Positionwise(LayerNorm(
        config["hidden_size"]
    ))

    embed = CompositeEmbedding(
        tok=tok_emb,
        pe=posi_emb,
        segment=seg_emb,
        postprocessor=emb_post
    )



end
