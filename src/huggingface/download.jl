using Pkg.Artifacts

using HTTP

const ARTIFACTS_TOML = joinpath(@__DIR__, "Artifacts.toml")

const CLOUDFRONT_DISTRIB_PREFIX = "https://cdn.huggingface.co"
const S3_BUCKET_PREFIX = "https://s3.amazonaws.com/models.huggingface.co/bert"
const DEFAULT_CONFIG_NAME = "config.json"
const DEFAULT_WEIGHT_NAME = "pytorch_model.bin"

get_registered_config_path(model_name; config=DEFAULT_CONFIG_NAME) = joinpath(get_registered_file_dir(joinpath(model_name, config)), config)
get_registered_weight_path(model_name; weight=DEFAULT_WEIGHT_NAME) = joinpath(get_registered_file_dir(joinpath(model_name, weight)), weight)
function get_registered_file_dir(name)
  global ARTIFACTS_TOML
  hash = artifact_hash(name, ARTIFACTS_TOML)
  isnothing(hash) && error("$name not registered.")
  artifact_path(hash)
end

get_or_download_hgf_config(model_name; config=DEFAULT_CONFIG_NAME) = get_or_download_hgf_file(model_name, config)
get_or_download_hgf_weight(model_name; weight=DEFAULT_WEIGHT_NAME) = get_or_download_hgf_file(model_name, weight)
function get_or_download_hgf_file(model_name, file_name)
  global CLOUDFRONT_DISTRIB_PREFIX
  hash = find_or_register_hgf_file_hash(CLOUDFRONT_DISTRIB_PREFIX, model_name, file_name)
  return joinpath(artifact_path(hash), file_name)
end

isregistered(name) = (global ARTIFACTS_TOML; !isnothing(artifact_hash(name, ARTIFACTS_TOML)))

isurl(path::AbstractString) = startswith(path, "http://") || startswith(path, "https://")

find_or_register_hgf_config_hash(path, model_name; config=DEFAULT_CONFIG_NAME) = find_or_register_hgf_file_hash(path, model_name, config)
find_or_register_hgf_weight_hash(path, model_name; weight=DEFAULT_WEIGHT_NAME) = find_or_register_hgf_file_hash(path, model_name, weight)
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
      filegetter = Base.Fix1(HTTP.download, url)
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

    # register
    bind_artifact!(ARTIFACTS_TOML, entry_name, file_hash)
  end

  file_hash
end
