using Test
using TimerOutputs
using Statistics
using ArgParse

const allowed_subject = (
    "all",          # test all subject
    "none",         # only test the existence of the model and the python packages installation
    "based_model",  # test the transformer backbone (without the task specific head)
    "task_head",    # test the whole model with task specific head
    "tokenizer",    # test the tokenizer and encoder, must specify corpus
    "whole_model",  # test tokenizer + task_head
)

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--subject", "-s"
            help = "a specific testing subject, should be one of $allowed_subject"
            arg_type = String
            default = "all"
        "--number", "-n"
            help = "the number of random sample for testing the model"
            arg_type = Int
            default = 100
        "--max-error"
            help = "the error bound for the square error"
            arg_type = Float64
            default = 0.1
        "--mean-error"
            help = "the error bound for the mean square error"
            arg_type = Float64
            default = 0.01
        "--output", "-o"
            help = "file name where the failed sample would be write to (for testing tokenizer only)."
            default = nothing
        "name"
            help = "model name"
            required = true
        "corpus"
            help = "corpus for testing, must specified if you are testing with tokenizer"
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
const output_file = args["output"]

const max_error = args["max-error"]
const mean_error = args["mean-error"]

if subject ∉ allowed_subject
    error("unknown testing subject: $subject")
end

if subject == "tokenizer" || subject == "all"
    isnothing(output_file) ||
        @assert !isfile(output_file) "File $output_file already exists"
end

if isnothing(corpus) && (subject == "tokenizer" || subject == "whole_model")
    error("testing tokenizer but no corpus provided.")
end

const to = TimerOutput()

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
include("whole_model.jl")
try
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
            cfg = HuggingFace.load_config(model_name)
            cfg = HuggingFace.HGFConfig(cfg; layer_norm_eps = 1e-9, layer_norm_epsilon = 1e-9)
            if cfg.model_type == "clip"
                if haskey(cfg, "text_config")
                    cfg = HuggingFace.HGFConfig(
                        cfg; text_config = HuggingFace.HGFConfig(cfg.text_config;
                                                                 layer_norm_eps = 1e-9, layer_norm_epsilon = 1e-9))
                end
                if haskey(cfg, "vision_config")
                    cfg = HuggingFace.HGFConfig(
                        cfg; vision_config = HuggingFace.HGFConfig(cfg.vision_config;
                                                                   layer_norm_eps = 1e-9, layer_norm_epsilon = 1e-9))
                end
            end
            cfg
        end "Failed to load configure file in Julia, probably unsupported"
        @info "Load configure file in Python"
        global pyconfig = @tryrun begin
            cfg = hgf_trf.AutoConfig.from_pretrained(model_name, layer_norm_eps = 1e-9, layer_norm_epsilon = 1e-9)
            if cfg.model_type == "clip"
                if haskey(cfg, "text_config")
                    cfg.text_config.layer_norm_eps = 1e-9
                    cfg.text_config.layer_norm_epsilon = 1e-9
                end
                if haskey(cfg, "vision_config")
                    cfg.vision_config.layer_norm_eps = 1e-9
                    cfg.vision_config.layer_norm_epsilon = 1e-9
                end
            end
            cfg
        end "Failed to load configure file in Python, probably unsupported"

        global vocab_size = if haskey(config, :vocab_size)
            config.vocab_size
        else
            @warn "Configure file doesn't have vocab_size information. Use 100 as a default value"
            100
        end

        if subject != "none"
            (subject == "all" || subject == "based_model") &&
                @timeit to "based_model" test_based_model(model_name, num; max_error, mean_error)
            (subject == "all" || subject == "task_head") &&
                @timeit to "task_head" test_task_head(model_name, num; max_error, mean_error)

            !isnothing(corpus) && (subject == "all" || subject == "tokenizer") &&
                @timeit to "tokenizer" test_tokenizer(model_name, corpus; output = output_file)
            !isnothing(corpus) && (subject == "all" || subject == "whole_model") &&
                @timeit to "whole_model" test_whole_model(model_name, corpus; max_error, mean_error)
        end
    end
catch e
    rethrow(e)
finally
    show(to)
    println()
end
