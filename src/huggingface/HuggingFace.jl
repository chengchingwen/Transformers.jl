module HuggingFace

using ..Transformers

using HuggingFaceApi

export @hgf_str,
    load_config,
    load_model,
    load_tokenizer,
    load_state_dict,
    load_hgf_pretrained

include("./utils.jl")
include("./download.jl")
include("./weight.jl")
include("./configs/config.jl")
include("./models/models.jl")
include("./tokenizer/tokenizer.jl")
include("./implementation/implement.jl")

"""
    `hgf"<model-name>:<item>"`

Get `item` from `model-name`. This will ensure the required data are downloaded. `item` can be "config",
 "tokenizer", and model related like "Model", or "ForMaskedLM", etc. Use [`get_model_type`](@ref) to see what
 model/task are supported.
"""
macro hgf_str(name)
  :(load_hgf_pretrained($(esc(name))))
end

"""
  `load_hgf_pretrained(name)`

The underlying function of [`@hgf_str`](@ref).
"""
function load_hgf_pretrained(name; kw...)
    name_item = rsplit(name, ':'; limit=2)
    all = length(name_item) == 1
    model_name, item = if all
        name, "model"
    else
        Iterators.map(String, name_item)
    end
    item = lowercase(item)

    cfg = load_config(model_name; kw...)
    item == "config" && return cfg

    (item == "tokenizer" || all) &&
        (tkr = load_tokenizer(model_name; config = cfg, kw...))
    item == "tokenizer" && return tkr

    model = load_model(cfg.model_type, model_name, item; config=cfg, kw...)

    if all
        return tkr, model
    else
        return model
    end
end

end
