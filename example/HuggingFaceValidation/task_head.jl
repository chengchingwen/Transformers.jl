function test_task_head(name, n; max_error = 1e-1, mean_error = 1e-2)
    global torch, hgf_trf, config, pyconfig, vocab_size
    @info "Validate $name task head"
    @testset "Task Head" begin
        @info "Found task type in confingure file" tasks = config.architectures
        model_type = config.model_type |> Symbol |> Val

        for task_type in config.architectures
            task_type = task_type_map(task_type)
            @info "Loading model with $task_type head in Python"
            hgf_model = @tryrun begin
                pyauto = hgf_trf[task_type]
                pyauto.from_pretrained(name, config = pyconfig)
            end "Failed to load model in Python"
            @info "Python model loaded successfully"

            @info "Loading model with $task_type head in Julia"
            model = @tryrun begin
                jl_task_type = chop(task_type, head=length(config.model_type), tail=0) |> lowercase |> Symbol |> Val
                jl_model_cons = HuggingFace.get_model_type(model_type, jl_task_type)
                HuggingFace.load_model(jl_model_cons, name; config)
            end "Failed to load the model in Julia"
            @info "Julia model loaded successfully"

            @info "Testing: $task_type head forward"
            for i = 1:n
                len = rand(50:100)
                indices = rand(1:vocab_size, len)
                pyindices = torch.tensor(indices .- 1).reshape(1, len)
                if HuggingFace.is_seq2seq(model)
                    len2 = rand(50:100)
                    indices2 = rand(1:vocab_size, len2)
                    pyindices2 = torch.tensor(indices2 .- 1).reshape(1, len2)
                    py_states = hgf_model(input_ids = pyindices, decoder_input_ids = pyindices2)
                    jl_states = model(indices, indices2)
                else
                    py_states = hgf_model(pyindices)
                    jl_states = model(reshape(indices, len, 1))
                end

                if haskey(jl_states, :logits)
                    py_result = rowmaj2colmaj(py_states["logits"].detach().numpy())
                    jl_result = jl_states.logits
                elseif haskey(jl_states, :prediction_logits)
                    py_result = rowmaj2colmaj(py_states["prediction_logits"].detach().numpy())
                    jl_result = jl_states.prediction_logits
                elseif haskey(jl_states, :start_logits)
                    py_result = rowmaj2colmaj(py_states["start_logits"].detach().numpy())
                    jl_result = jl_states.start_logits
                end
                jl_result = reshape(jl_result, Val(ndims(py_result)))
                diff = (py_result .- jl_result) .^ 2
                @test maximum(diff) < max_error
                @test mean(diff) < mean_error
            end
        end
    end
end

const TASK_TYPE_MAP = Dict{String, String}(
    "T5WithLMHeadModel" => "T5ForConditionalGeneration",
)

function task_type_map(task_type)
    global TASK_TYPE_MAP
    new_task_type = get(TASK_TYPE_MAP, task_type, task_type)
    if new_task_type != task_type
        @info "$task_type mapped to $new_task_type"
    end
    return new_task_type
end
