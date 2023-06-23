function test_whole_model(name, corpus; max_error = 1e-1, mean_error = 1e-2)
    global torch, hgf_trf, config, pyconfig, vocab_size
    global error_samples = []
    @info "Validate $name whole model with corpus $corpus"
    @testset "Whole model" begin
        model_type = config.model_type |> Symbol
        task_type = first(config.architectures)
        isfile(corpus) || error("corpus file $corpus do not exist.")

        @info "Loading tokenizer in Python"
        global hgf_tkr = @tryrun begin
            hgf_trf.AutoTokenizer.from_pretrained(name, config = pyconfig)
        end "Failed to load the tokenizer in Python"
        @info "Python tokenizer loaded successfully"

        @info "Loading tokenizer in Julia"
        global tkr = @tryrun begin
            HuggingFace.load_tokenizer(name; config)
        end "Failed to load the tokenizer in Julia"
        @info "Julia tokenizer loaded successfully"

        @info "Loading model with $task_type head in Python"
        global hgf_model = @tryrun begin
            pyauto = hgf_trf[task_type]
            pyauto.from_pretrained(name, config = pyconfig)
        end "Failed to load model in Python"
        @info "Python model loaded successfully"

        @info "Loading model with $task_type head in Julia"
        global model = @tryrun begin
            jl_task_type = chop(task_type, head=length(config.model_type) - count('_', config.model_type), tail=0) |> lowercase |> Symbol
            HuggingFace.load_model(model_type, name, jl_task_type; config)
        end "Failed to load the model in Julia"
        @info "Julia model loaded successfully"

        @info "Testing: Whole model"
        fd = nothing
        sum_hidden_avg_err = 0.0
        sum_hidden_max_err = 0.0
        sum_avg_err = 0.0
        sum_max_err = 0.0
        len = 0
        n_tokens = 0
        try
            fd = open(corpus)
            for line in eachline(fd)
                isempty(line) && continue
                len += 1
                jl_tokens = TextEncodeBase.getvalue.(TextEncodeBase.tokenize(tkr, line))
                py_tokens = hgf_tkr.tokenize(line)
                @test jl_tokens == py_tokens
                jl_indices = collect(reinterpret(Int32, encode(tkr, line).token))
                _py_indices = collect(hgf_tkr(line)["input_ids"])
                py_indices = _py_indices .+ 1
                @test jl_indices == py_indices

                if HuggingFace.is_seq2seq(model)
                    n_tokens += 2 * length(py_tokens)
                    py_input = torch.tensor(_py_indices).reshape(1, length(py_indices))
                    jl_input = encode(tkr, line, line)
                    py_states = hgf_model(input_ids = py_input, decoder_input_ids = py_input,
                                          output_hidden_states = true)
                    jl_states = model(jl_input)
                    py_hidden = rowmaj2colmaj(py_states["decoder_hidden_states"][end].detach().numpy())
                else
                    n_tokens += length(py_tokens)
                    py_input = torch.tensor(_py_indices).reshape(1, length(py_indices))
                    jl_input = encode(tkr, line)
                    py_states = hgf_model(py_input, output_hidden_states = true)
                    jl_states = model(jl_input)
                    py_hidden = rowmaj2colmaj(py_states["hidden_states"][end].detach().numpy())
                end
                jl_hidden = jl_states.hidden_state
                diff_hidden = (py_hidden .- jl_hidden) .^ 2
                avg_hidden_err = mean(diff_hidden)
                max_hidden_err = maximum(diff_hidden)
                sum_hidden_avg_err += avg_hidden_err
                sum_hidden_max_err = max(max_hidden_err, sum_hidden_max_err)
                @test avg_hidden_err < mean_error
                @test max_hidden_err < max_error

                if haskey(py_states, "logits")
                    py_result = rowmaj2colmaj(py_states["logits"].detach().numpy())
                    jl_result = jl_states.logit
                elseif haskey(py_states, "prediction_logits")
                    py_result = rowmaj2colmaj(py_states["prediction_logits"].detach().numpy())
                    jl_result = jl_states.logit
                elseif haskey(py_states, "start_logits")
                    py_result = rowmaj2colmaj(py_states["start_logits"].detach().numpy())
                    jl_result = jl_states.start_logit
                end
                jl_result = reshape(jl_result, Val(ndims(py_result)))
                diff = (py_result .- jl_result) .^ 2
                avg_err = mean(diff)
                max_err = maximum(diff)
                sum_avg_err += avg_err
                sum_max_err = max(max_err, sum_max_err)
                @test avg_err < mean_error
                @test max_err < max_error

                jl_pred = Flux.onecold(jl_result)
                py_pred = Flux.onecold(py_result)
                @test jl_pred == py_pred
                if jl_pred != py_pred
                    push!(error_samples, (line, (jl_input, jl_pred), (py_input, py_pred)))
                end
            end
        catch e
            isnothing(fd) || close(fd)
            rethrow(e)
        end
        @info "average hidden_state error" max = sum_hidden_max_err mean = sum_hidden_avg_err / len
        @info "average logit error" max = sum_max_err mean = sum_avg_err / len
        @info "average tokens per sample" token = n_tokens / len
    end
end

function show_lm_head_error_samples(error_samples)
    global tkr
    for (line, jl, py) in error_samples
        jl_word = lookup(tkr.vocab, jl[2])
        py_word = lookup(tkr.vocab, py[2])
        _diff = jl_word .!= py_word
        @show jl_word[_diff]
        @show py_word[_diff]
    end
end
