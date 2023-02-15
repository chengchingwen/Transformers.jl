"""
    load_config(model_name; local_files_only = false, cache = true)

Load the configuration file of `model_name` from huggingface hub. By default, this function would check if `model_name`
 exists on huggingface hub, download the configuration file (and cache it if `cache` is set), and then load and return
 the config`::HGFConfig`. If `local_files_only = false`, it would check whether the configuration file is up-to-date
 and update if not (and thus require network access every time it is called). By setting `local_files_only = true`, it
 would try to find the files from the cache directly and error out if not found. For managing the caches, see the
 `HuggingFaceApi.jl` package. This function would require the configuration file has a field about the `model_type`, if
 not, use `load_config(model_type, HuggingFace.load_config_dict(model_name; local_files_only, cache))` with `model_type`
 manually provided.

See also: [`HGFConfig`](@ref)

# Example

```julia-repl
julia> load_config("bert-base-cased")
Transformers.HuggingFace.HGFConfig{:bert, JSON3.Object{Vector{UInt8}, Vector{UInt64}}, Nothing} with 19 entries:
  :architectures                => ["BertForMaskedLM"]
  :attention_probs_dropout_prob => 0.1
  :gradient_checkpointing       => false
  :hidden_act                   => "gelu"
  :hidden_dropout_prob          => 0.1
  :hidden_size                  => 768
  :initializer_range            => 0.02
  :intermediate_size            => 3072
  :layer_norm_eps               => 1.0e-12
  :max_position_embeddings      => 512
  :model_type                   => "bert"
  :num_attention_heads          => 12
  :num_hidden_layers            => 12
  :pad_token_id                 => 0
  :position_embedding_type      => "absolute"
  :transformers_version         => "4.6.0.dev0"
  :type_vocab_size              => 2
  :use_cache                    => true
  :vocab_size                   => 28996

```
"""
load_config(model; kw...) = _load_config(load_config_dict(model; kw...))

load_config_dict(model_name; kw...) = JSON3.read(read(hgf_model_config(model_name; kw...)))

_load_config(cfg_file::AbstractString) = _load_config(JSON3.read(read(cfg_file)))
function _load_config(cfg)
    !haskey(cfg, :model_type) && error(
        """
        \"model_type\" not specified in config file.
        Use `load_config_dict` to load the content and specify the model type with `load_config`.
        E.g. `load_config(:bert, load_config_dict(model_name))`
        """
    )
    model_type = Symbol(cfg[:model_type])
    return load_config(model_type, cfg)
end

@valsplit config_type(Val(model_type::Symbol)) = HGFConfig{model_type}

"""
    load_config(model_type, cfg)

Load `cfg` as `model_type`. This is used for manually load a config when `model_type` is not specified in the config.
 `model_type` is a `Symbol` of the model type like `:bert`, `:gpt2`, `:t5`, etc.
"""
load_config(model_type::Union{Symbol, Val}, cfg) = load_config(config_type(model_type), cfg)

function load_config(model_type::Type{<:HGFConfig}, cfg)
    overwrite = Dict{Symbol, Any}()
    haskey(cfg, :id2label) && !isnothing(cfg.id2label) &&
        (overwrite[:id2label] = Dict{Int, String}(parse(Int, String(key))=>val for (key, val) in cfg.id2label))
    haskey(cfg, :label2id) && !isnothing(cfg.label2id) &&
        (overwrite[:label2id] = Dict{String, Int}(String(key)=>val for (key, val) in cfg.label2id))
    return model_type(cfg, overwrite)
end

"""
    save_config(model_name, config; path = pwd(), config_name = CONFIG_NAME, force = false)

Save the `config` at `<path>/<model_name>/<config_name>`. This would error out if the file already exists but `force`
 not set.
"""
function save_config(model_name, config; path = pwd(), config_name = CONFIG_NAME, force = false)
    model_path = joinpath(path, model_name)
    mkpath(model_path)
    config_file = joinpath(model_path, config_name)
    if isfile(config_file)
        if force
            @warn "$config_file forcely updated."
        else
            error("$config_file already exists. set `force = true` for force update.")
        end
    end
    open(config_file, "w+") do io
        JSON3.write(io, config)
    end
    return config_file
end
