using ArgParse
using Transformers

using Random
Random.seed!(0)

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--gpu", "-g"
        help = "use gpu"
        action = :store_true

        "task"
        help = "task name"
        required = true
        range_tester = x-> x ∈ ["wmt14", "iwslt2016", "copy"]
    end

    return parse_args(ARGS, s)
end

const args = parse_commandline()

enable_gpu(args["gpu"])

const task = args["task"]

include(joinpath(@__DIR__, task, "train.jl"))
