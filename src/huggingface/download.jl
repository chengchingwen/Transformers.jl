using Pkg.Artifacts
using SHA

using HTTP

const ARTIFACTS_TOML = joinpath(@__DIR__, "Artifacts.toml")

const CLOUDFRONT_DISTRIB_PREFIX = "https://cdn.huggingface.co"
const S3_BUCKET_PREFIX = "https://s3.amazonaws.com/models.huggingface.co/bert"
const DEFAULT_CONFIG_NAME = "config.json"
const DEFAULT_WEIGHT_NAME = "pytorch_model.bin"

"""
Workflow:
1. Pretrained model is from huggingface (yes=>a. / no=>b.)

a. Already use the model in python before. (yes=>a-1. / no=>a-2.)

a-1. There is a cached pretrain file on the computer. Use [`get_or_download_hgf_file`](@ref) directly. This will copy
the file from cached dir to our Artifacts dir and register on Artifacts.toml.
a-2. No cached files. Also use [`get_or_download_hgf_file`](@ref) directly. This will download the file from 
huggingface's model server to our Artifacts dir and register on Artifact.toml.

b. Using the local pretrained files. Use [`find_or_register_hgf_file_hash`](@ref) to register the file to our 
Artifacts system. Once the file is registered, you can find the entry appear on Artifacts.toml.

2. Once the model is managed under Julia's Artifacts system. we can use either [`get_registered_file_dir`](@ref) or 
[`get_or_download_hgf_file`](@ref) to get the pretrain file or dir.
"""

"""
  `get_registered_config_path(model_name; config=DEFAULT_CONFIG_NAME)`

Get the config path of a given `model_name` which should be already registered on `Artifacts.toml`.
"""
get_registered_config_path(model_name; config=DEFAULT_CONFIG_NAME) = joinpath(get_registered_file_dir(joinpath(model_name, config)), config)

"""
  `get_registered_weight_path(model_name; weight=DEFAULT_WEIGHT_NAME)`

Get the weight path of a given `model_name` which should be already registered on `Artifacts.toml`.
"""
get_registered_weight_path(model_name; weight=DEFAULT_WEIGHT_NAME) = joinpath(get_registered_file_dir(joinpath(model_name, weight)), weight)

"""
  `get_registered_file_dir(name)`

Get the dir path of a given `name` which should be already registered on `Artifacts.toml`.

See also: [`get_registered_config_path`](@ref), [`get_registered_weight_path`](@ref)
"""
function get_registered_file_dir(name)
  global ARTIFACTS_TOML
  hash = artifact_hash(name, ARTIFACTS_TOML)
  isnothing(hash) && error("$name not registered.")
  artifact_path(hash)
end

"""
  `get_or_download_hgf_config(model_name; config=DEFAULT_CONFIG_NAME)`

get the config file path of the given `model_name` and `file_name`

See also: [`get_or_download_hgf_file`](@ref)
"""
get_or_download_hgf_config(model_name; config=DEFAULT_CONFIG_NAME) = get_or_download_hgf_file(model_name, config)

"""
  `get_or_download_hgf_weight(model_name; weight=DEFAULT_WEIGHT_NAME)`

get the weight file path of the given `model_name` and `file_name`

See also: [`get_or_download_hgf_file`](@ref)
"""
get_or_download_hgf_weight(model_name; weight=DEFAULT_WEIGHT_NAME) = get_or_download_hgf_file(model_name, weight)

"""
  `get_or_download_hgf_file(model_name, file_name)`

get the file path of the given `model_name` and `file_name`. Automatically download and register from 
huggingface server if file not found on `Artifacts.toml`. To use with a local file, register with 
[`find_or_register_hgf_file_hash`](@ref) first.
"""
function get_or_download_hgf_file(model_name, file_name)
  global CLOUDFRONT_DISTRIB_PREFIX
  hash = find_or_register_hgf_file_hash(CLOUDFRONT_DISTRIB_PREFIX, model_name, file_name)
  return joinpath(artifact_path(hash), file_name)
end

isregistered(name) = (global ARTIFACTS_TOML; !isnothing(artifact_hash(name, ARTIFACTS_TOML)))

isurl(path::AbstractString) = startswith(path, "http://") || startswith(path, "https://")

"""
  `find_or_register_hgf_config_hash(path, model_name; config=DEFAULT_CONFIG_NAME)`

Get the artifacts hash of config of the give model.

See also: [`find_or_register_hgf_file_hash`](@ref)
"""
find_or_register_hgf_config_hash(path, model_name; config=DEFAULT_CONFIG_NAME) = find_or_register_hgf_file_hash(path, model_name, config)

"""
  `find_or_register_hgf_weight_hash(path, model_name; weight=DEFAULT_WEIGHT_NAME)`

Get the artifacts hash of weight of the give model.

See also: [`find_or_register_hgf_file_hash`](@ref)
"""
find_or_register_hgf_weight_hash(path, model_name; weight=DEFAULT_WEIGHT_NAME) = find_or_register_hgf_file_hash(path, model_name, weight)

"""
  `find_or_register_hgf_file_hash(path, model_name, file_name)`

Get the artifacts hash of the give `<model_name>/<file_name>`. If not found in Artifacts.toml, get the file from `path` 
and register on Artifacts.toml. `path` can be either a url or a local path.
"""
function find_or_register_hgf_file_hash(path, model_name, file_name)
  global ARTIFACTS_TOML
  entry_name = joinpath(model_name, file_name)
  file_hash = artifact_hash(entry_name, ARTIFACTS_TOML)

  if isnothing(file_hash) || !artifact_exists(file_hash) # not found, try get model and register
    if isurl(path) # path is a url, download with huggingface url format
      islegacy = !occursin('/', model_name)
      url = islegacy ?
        join((path, model_name, file_name), '/', '-') :
        joinpath(path, entry_name)
      filegetter = Base.Fix1(hgf_file_download, url)
    else # path is a local file, cp to artifact dir
      file_path = joinpath(path, file_name)
      isfile(file_path) || error("Can't register to $entry_name: file $file_path not found.")
      filegetter = function (dest)
        cp(file_path, dest)
      end
    end

    file_hash = create_artifact() do artifact_dir
        filegetter(joinpath(artifact_dir, file_name))
    end

    # register if no in artifact.toml
    if isnothing(artifact_hash(entry_name, ARTIFACTS_TOML))
      bind_artifact!(ARTIFACTS_TOML, entry_name, file_hash)
    end
  end

  file_hash
end

"Get the cache dir used by huggingface/transformers"
function get_th_hgf_cache_dir()
  return get(ENV, "TRANSFORMERS_CACHE",
             get(ENV, "PYTORCH_TRANSFORMERS_CACHE",
                 get(ENV, "PYTORCH_PRETRAINED_BERT_CACHE",
                     joinpath(
                       get(ENV, "TORCH_HOME", get(ENV, "XDG_CACHE_HOME", "~/.cache")),
                       "torch", "transformers")))) |> expanduser
end

"Find the huggingface cached files from their cache dir"
function get_hgf_cached_file(url; cache_dir=get_th_hgf_cache_dir())
  isdir(cache_dir) || return nothing
  global CLOUDFRONT_DISTRIB_PREFIX, S3_BUCKET_PREFIX
  iscdn = startswith(url, CLOUDFRONT_DISTRIB_PREFIX)
  iss3 = startswith(url, S3_BUCKET_PREFIX)

  hashstr(x) = bytes2hex(sha256(x))
  match(hash) = filter(f->startswith(f, hash) && !endswith(f, ".json") && !endswith(f, ".lock"),
                       readdir(cache_dir))

  hash = hashstr(url)
  mf = match(hash)
  !isempty(mf) && return joinpath(cache_dir, last(mf)) # if found then return

  if iscdn # using cdn, retry with s3
    alter_url = replace(url, CLOUDFRONT_DISTRIB_PREFIX=>S3_BUCKET_PREFIX)
  elseif iss3 # using s3, retry with cdn
    alter_url = replace(url, S3_BUCKET_PREFIX=>CLOUDFRONT_DISTRIB_PREFIX)
  else # custom url
    return nothing
  end

  # retry with alternative url
  hash = hashstr(alter_url)
  mf = match(hash)

  if !isempty(mf)
    return joinpath(cache_dir, last(mf))
  else
    return nothing
  end
end

"Get files from huggingface. Copy if found in cache dir, otherwise download from their server."
function hgf_file_download(url, dest)
  cached_file = get_hgf_cached_file(url)
  file_name = basename(dest)
  if isnothing(cached_file) # not found, just download
    @info "No local $file_name found. downloading..."
    HTTP.download(url, dest)
  else # found, copy from cache
    @info "PyTorch cached $file_name found. copying..."
    cp(cached_file, dest)
  end
end
