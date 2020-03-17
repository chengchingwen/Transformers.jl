using Pkg.TOML
using Markdown

const all_config = open(TOML.parse, joinpath(@__DIR__, "pretrains.toml"))

function description(config::Dict)
  desc = config["description"]
  host = config["host"]
  link = config["link"]
  if haskey(config, "cite")
    cite = config["cite"]
    description(desc, host, link, cite)
  else
    description(desc, host, link)
  end
end
function description(description::String, host::String, link::String, cite=nothing)
  """
  $description
  Released by $(host) at $(link).
  $(isnothing(cite) ? "" : "\nCiting:\n$cite")"""
end

subword_type(config::Dict) = subword_type(Val(Symbol(config["subword"])), config)
function subword_type(::Val{:bpe}, config)
  vocab = config["vocab"]
  case = config["case"]
  "subword with $(vocab)-vocabularies-sized $(case) bpe tokenizer."
end
function subword_type(::Val{:wordpiece}, config)
  case = config["case"]
  "subword with google wordpiece $case tokenizer."
end

function model_summary(mt::String, config::Dict)
  head = config["head"]
  hidden = config["hidden"]
  layer = config["layer"]
  tk_desc = subword_type(config)
  desc = get(config, "desc", nothing)
  """
  $mt model with $(layer) layers, $(hidden) hidden units, and $(head) attention heads.
  $tk_desc$(isnothing(desc) ? "" : "\n$desc" )"""
end

function register_config(configs)
  for (model_type, model_config) in pairs(configs)
    models = filter(!isequal("description"), keys(model_config))
    mt_desc = model_config["description"]
    for model in models
      config = model_config[model]
      model_configs = config["models"]
      model_desc = description(config)

      for (model_name, model_detail) in model_configs
        summary = model_summary(model_type, model_detail)
        depname = "$(uppercase(model_type))-$(model_name)"
        depdesc = join([mt_desc, summary, model_desc], "\n\n")
        url = model_detail["url"]
        checksum = model_detail["checksum"]
        dep = DataDep(depname, depdesc, url, checksum; fetch_method=download_gdrive)
        DataDeps.register(dep)
      end
    end
  end
end


match(model, name, query) = isequal(lowercase(query), model) || isequal(query, name)

"""
  pretrains(query::String = ""; detailed::Bool = false)

Show all available models. you can also query a specific `model` or `model name`.
show more detail with `detailed = true`.
"""
function pretrains(query::String = ""; detailed::Bool = false)
  global all_config
  rows = [Any["Type", "model", "model name", "support items"]]
  detailed &&
    push!(rows[1], "detail description")

  for (model_type, model_config) in pairs(all_config)
    models = filter(!isequal("description"), keys(model_config))
    mt_desc = model_config["description"]
    for model in models
      config = model_config[model]
      model_configs = config["models"]
      model_desc = description(config)

      for (model_name, model_detail) in model_configs
        summary = model_summary(model_type, model_detail)
        name = model_detail["name"]
        items = join(model_detail["items"], ", ")
        if isempty(query) || match(model, name, query)
          row = Any[uppercasefirst(model_type), model, name, items]
        else
          continue
        end
        detailed &&
          push!(row, summary)

        push!(rows, row)
      end
    end
  end
  Markdown.MD(Any[Markdown.Table(rows, detailed ? [:l,:l,:l,:l,:l] : [:l,:l,:l,:l])])
end

function parse_model(str)
  m = Base.match(r"([a-zA-Z]+)[-:]([a-zA-Z\d\-_]+):?(\S*)", str)
  type, name, item = m.captures
  item = isempty(item) ? :all : Symbol(item)
  return lowercase(type), name, item
end
