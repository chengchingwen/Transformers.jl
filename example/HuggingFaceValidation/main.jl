using Test
using Statistics
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--subject", "-s"
            help = "a specific testing subject"
            arg_type = String
            default = "all"

        "--number", "-n"
            help = "the number of random sample for testing the model"
            arg_type = Int
            default = 100
        "name"
            help = "model name"
            required = true
        "corpus"
            help = "corpus for testing tokenizer"
            required = false
            default = nothing
    end
    return parse_args(ARGS, s)
end
const args = parse_commandline()
const model_name = args["name"]
const subject = args["subject"] |> lowercase
const num = args["number"]
const corpus = args["corpus"]

if subject ∉ ("all", "based_model", "task_head", "tokenizer")
    error("unknown testing subject: $subject")
end

if isnothing(corpus) && subject == "tokenizer"
    error("testing tokenizer but no corpus provided.")
end

using PyCall
using HuggingFaceApi
using Flux
using TextEncodeBase
using Transformers
using Transformers.HuggingFace

include("utils.jl")
include("based_model.jl")
include("task_head.jl")
include("tokenizer.jl")

@testset "HuggingFaceValiation" begin
    @tryrun begin
        HuggingFaceApi.model_info(model_name)
    end "Cannot find $model_name: Does this model really exist on huginggface hub?"

    @info "Loading python packages"
    global torch = @tryrun begin
        pyimport("torch")
    end "Importing pytorch result in error. Make sure pytorch is installed correctly."
    global hgf_trf = @tryrun begin
        pyimport("transformers")
    end "Importing huggingface transformers result in error. Make sure transformers is installed correctly."
    @info "Python packages loaded successfully"

    @info "Load configure file in Julia"
    global config = @tryrun begin
        HuggingFace.load_config(model_name)
    end "Failed to load configure file in Julia, probably unsupported"

    global vocab_size = if haskey(config, :vocab_size)
        config.vocab_size
    else
        @warn "Configure file doesn't have vocab_size information. Use 100 as a default value"
        100
    end

    (subject == "all" || subject == "based_model") && test_based_model(model_name, num)
    (subject == "all" || subject == "task_head") && test_task_head(model_name, num)

    !isnothing(corpus) && (subject == "all" || subject == "tokenizer") && test_tokenizer(model_name, corpus)
end
