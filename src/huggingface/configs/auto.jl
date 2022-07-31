"""
    load_config(model_name)

Get the configuration file of given model.
"""
load_config(model; kw...) = _load_config(load_config_dict(model; kw...))

load_config_dict(model_name; kw...) = json_load(hgf_model_config(model_name; kw...))

_load_config(cfg_file::AbstractString) = _load_config(json_load(cfg_file))
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

@valsplit config_type(Val(model_type::Symbol)) = error("Unknown model type: $model_type")

"""
    load_config(model_type, cfg)

Load `cfg` as `model_type`. This is used for manually load a config when `model_type` is not specified in the config.
"""
load_config(model_type::Union{Symbol, Val}, cfg) = load_config(config_type(model_type), cfg)
load_config(cfg_type::Type, cfg) = cfg_type(; cfg...)

"""
    save_config(model_name, config; path = pwd(), config_name = CONFIG_NAME)

save the `config` at `<path>/<model_name>/<config_name>`.
"""
function save_config(model_name, config; path = pwd(), config_name = CONFIG_NAME)
    model_path = joinpath(path, model_name)
    !isdirpath(model_path) && error("$model_path is not a dir.")
    mkpath(model_path)
    config_file = joinpath(model_path, config_name)
    open(config_file, "w+") do io
        JSON.print(io, config)
    end
    return config_file
end

