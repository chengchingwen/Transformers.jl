module Pretrain

using Markdown

using Flux
using BSON
using DataDeps
using ZipFile

using ..Transformers: BidirectionalEncoder, GenerativePreTrain
using ..Basic
using ..Datasets: download_gdrive

export @pretrain_str, load_pretrain, pretrains


isbon(s) = endswith(s, ".bson")
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

macro pretrain_str(name)
    :(load_pretrain($(esc(name))))
end

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
        BidirectionalEncoder.load_bert_pretrain(model_path, item)
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
        GenerativePreTrain.load_gpt_pretrain(model_path, item; kw...)
    else
        error("unknown pretrain")
    end
end

function pretrains()
    rows = [Any["model", "model name", "support items"]]
    for (model, configs) ∈ zip(("GPT", "Bert"), (_get_gpt_config, _get_bert_config))
        for config ∈ values(configs)
            name = config[:name]
            items = join(config[:items], ", ")
            push!(rows, Any[model, name, items])
        end
    end
    Markdown.MD(Any[Markdown.Table(rows, [:l, :l, :l])])
end

end
