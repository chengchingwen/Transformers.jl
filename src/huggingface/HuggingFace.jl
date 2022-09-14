module HuggingFace

using ..Transformers

using HuggingFaceApi

export @hgf_str,
    load_config,
    load_model,
    load_model!,
    load_tokenizer,
    load_state_dict,
    load_state,
    save_config,
    save_model

include("./utils.jl")
include("./download.jl")
include("./weight.jl")
include("./configs/config.jl")
include("./models/models.jl")
include("./tokenizer/tokenizer.jl")
include("./implementation/implement.jl")

"""
  `hgf"<model-name>:<item>"`

Get `item` from `model-name`. This will ensure the required
data are downloaded and registered. `item` can be "config",
"tokenizer", and model related like "model", or "formaskedlm", etc. Use [`get_model_type`](@ref) to see what
model/task are supported.
"""
macro hgf_str(name)
  :(load_hgf_pretrained($(esc(name))))
end

"""
  `load_hgf_pretrained(name)`

The underlying function of [`@hgf_str`](@ref).
"""
function load_hgf_pretrained(name)
    name_item = rsplit(name, ':'; limit=2)
    all = length(name_item) == 1
    model_name, item = if all
        name, "model"
    else
        Iterators.map(String, name_item)
    end

    hgf_model_config(model_name)
    cfg = load_config(model_name)

    item == "config" &&
        return cfg

    (item == "tokenizer" || all) &&
        (tkr = load_tokenizer(model_name; config = cfg))

    item == "tokenizer" &&
        return tkr

    model_type = cfg.model_type
    model_cons = get_model_type((Val ∘ Symbol)(model_type), (Val ∘ Symbol ∘ lowercase)(item))

    hgf_model_weight(model_name)
    model = load_model(model_cons, model_name; config=cfg)

    if all
        tkr = load_tokenizer(model_name; config = cfg)
        return tkr, model
    else
        return model
    end
end

end
