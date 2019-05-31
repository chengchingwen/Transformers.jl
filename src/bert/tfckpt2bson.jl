# turn a tf bert format to bson

using JSON
using BSON
# using ZipFile
using TensorFlow

using Flux
using Flux: loadparams!

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
  

end

