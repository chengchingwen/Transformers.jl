using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--gpu", "-g"
            help = "use gpu"
            action = :store_true
        "task"
            help = "task name"
            required = true
            range_tester = x-> x ∈ ["cola", "mnli", "mrpc"]
    end

    return parse_args(ARGS, s)
end

const args = parse_commandline()

if args["gpu"]
    @eval using CuArrays
end
