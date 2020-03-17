module Pretrain

using Markdown

using Flux
using BSON
using DataDeps
using ZipFile

import ..Transformers
using ..Basic
using ..Datasets: download_gdrive

export @pretrain_str, load_pretrain, pretrains


isbson(s) = endswith(s, ".bson")
iszip(s) = endswith(s, ".zip")
isnpbson(s) = endswith(s, ".npbson")
istfbson(s) = endswith(s, ".tfbson")

zipname(z::ZipFile.Reader) = z.files[1].name
zipfile(z::ZipFile.Reader, name) = (idx = findfirst(zf->isequal(name)(zf.name), z.files)) !== nothing ? z.files[idx] : nothing
findfile(z::ZipFile.Reader, name) = zipfile(z, joinpath(zipname(z), name))

include("./config.jl")

function __init__()
  register_config(all_config)
end

"""
    pretrain"model-description:item"

convenient macro for loading data from pretrain. Use DataDeps to download automatically, if a model is not downlaod. the string should be in `pretrain"<model>-<model-name>:<item>"` format.

see also `Pretrain.pretrains()`.
"""
macro pretrain_str(name)
    :(load_pretrain($(esc(name))))
end

loading_method(::Val{:bert}) = Transformers.load_bert_pretrain
loading_method(::Val{:gpt}) = Transformers.load_gpt_pretrain
loading_method(x) = error("unknown pretrain type")

"""
    load_pretrain(name; kw...)

same as `@pretrain_str`, but can pass keyword argument if needed.
"""
function load_pretrain(str; kw...)
  type, name, item = parse_model(str)
  loader = loading_method(Val(Symbol(lowercase(type))))
  if type == "gpt"
    model_path = @datadep_str("$(uppercase(type))-$name/$(name).npbson")
  else
    model_path = @datadep_str("$(uppercase(type))-$name/$(name).tfbson")
  end
  loader(model_path, item; kw...)
end


end
