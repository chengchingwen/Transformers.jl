module HuggingFace

using ..Transformers

export get_or_download_hgf_file,
  get_or_download_hgf_config,
  get_or_download_hgf_weight,
  @hgf_str,
  load_config,
  load_model,
  load_model!,
  load_state_dict,
  load_state

include("./download.jl")
include("./weight.jl")
include("./configs/config.jl")
include("./models/models.jl")

macro hgf_str(name)
  :(load_hgf_pretrained($(esc(name))))
end

get_model_type(model, task) = error("Model $model doesn't support this kind of task: $task")

function load_hgf_pretrained(name)
  model_name, item = rsplit(name, ':'; limit=2)
  get_or_download_hgf_config(model_name)
  cfg = load_config(model_name)

  item == "config" &&
    return cfg

  item == "tokenizer" &&
    error("tokenizer not support yet.")

  model_type = cfg.model_type
  model_cons = get_model_type((Val ∘ Symbol ∘ lowercase)(model_type), (Val ∘ Symbol ∘ lowercase)(item))

  get_or_download_hgf_weight(model_name)
  model = load_model(model_cons, model_name; config=cfg)

  return model
end

end
