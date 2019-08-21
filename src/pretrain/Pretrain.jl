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

include("./gpt_pretrain.jl")
include("./bert_pretrain.jl")

function __init__()
    gpt_init()
    bert_init()
end


"""
    pretrain"model-description:item"

convenient macro for loading data from pretrain. Use DataDeps to download automatically, if a model is not downlaod. the string should be in `pretrain"<model>-<model-name>:<item>"` format.

see also `Pretrain.pretrains()`.
"""
macro pretrain_str(name)
    :(load_pretrain($(esc(name))))
end


"""
    load_pretrain(name; kw...)

same as `@pretrain_str`, but can pass keyword argument if needed.
"""
function load_pretrain(name; kw...)
    lowered = lowercase(name)
    if startswith(lowered, "bert")
        col = findlast(isequal(':'), name)
        if col === nothing
            model_name = name[6:end]
            item = :all
        else
            model_name = name[6:col-1]
            item = Symbol(name[col+1:end])
        end
        !haskey(_get_bert_config, model_name) && error("unknown bert pretrain name")
        model_path = @datadep_str("BERT-$model_name/$model_name.tfbson")
        Transformers.load_bert_pretrain(model_path, item)
    elseif startswith(lowered, "gpt") && lowered[4] != '2'
        col = findlast(isequal(':'), name)
        if col === nothing
            model_name = name[5:end]
            item = :all
        else
            model_name = name[5:col-1]
            item = Symbol(name[col+1:end])
        end
        !haskey(_get_gpt_config, model_name) && error("unknown gpt pretrain name")
        model_path = @datadep_str("GPT-$model_name/$model_name.npbson")
        Transformers.load_gpt_pretrain(model_path, item; kw...)
    else
        error("unknown pretrain")
    end
end


"""
    pretrains(model::String = "")

Show all available model.
"""
function pretrains(model::String = "")
    rows = [Any["model", "model name", "support items"]]
    if isempty(model)
        for (model, configs) ∈ zip(("GPT", "Bert"), (_get_gpt_config, _get_bert_config))
            for config ∈ values(configs)
                name = config[:name]
                items = join(config[:items], ", ")
                push!(rows, Any[model, name, items])
            end
        end
    else
        lowered_model = lowercase(model)
        for (model, configs) ∈ zip(("GPT", "Bert"), (_get_gpt_config, _get_bert_config))
            if lowered_model == lowercase(model)
                for config ∈ values(configs)
                    name = config[:name]
                    items = join(config[:items], ", ")
                    push!(rows, Any[model, name, items])
                end
            end
        end
    end
    Markdown.MD(Any[Markdown.Table(rows, [:l, :l, :l])])
end

end
