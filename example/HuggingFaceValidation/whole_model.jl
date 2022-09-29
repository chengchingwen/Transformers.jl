function test_whole_model(name, corpus; max_error = 1e-1, mean_error = 1e-2)
    global torch, hgf_trf, config, vocab_size
    global error_samples = []    
    @info "Validate $name whole model with corpus $corpus"
    @testset "Whole model" begin
        model_type = config.model_type |> Symbol |> Val
        task_type = first(config.architectures)
        isfile(corpus) || error("corpus file $corpus do not exist.")

        @info "Loading tokenizer in Python"
        global hgf_tkr = @tryrun begin
            hgf_trf.AutoTokenizer.from_pretrained(name)
        end "Failed to load the tokenizer in Python"
        @info "Python tokenizer loaded successfully"

        @info "Loading tokenizer in Julia"
        global tkr = @tryrun begin
            HuggingFace.load_hgf_pretrained("$name:tokenizer")
        end "Failed to load the tokenizer in Julia"
        @info "Julia tokenizer loaded successfully"

        @info "Loading model with $task_type head in Python"
        global hgf_model = @tryrun begin
            pyauto = hgf_trf[task_type]
            pyauto.from_pretrained(name)
        end "Failed to load model in Python"
        @info "Python model loaded successfully"

        @info "Loading model with $task_type head in Julia"
        global model = @tryrun begin
            jl_task_type = chop(task_type, head=length(config.model_type), tail=0) |> lowercase |> Symbol |> Val
            jl_model_cons = HuggingFace.get_model_type(model_type, jl_task_type)
            HuggingFace.load_model(jl_model_cons, name; config)
        end "Failed to load the model in Julia"
        @info "Julia model loaded successfully"

        @info "Testing: Whole model"
        fd = nothing
        try
            fd = open(corpus)
            for line in eachline(fd)
                isempty(line) && continue
                jl_tokens = TextEncodeBase.getvalue.(TextEncodeBase.tokenize(tkr, line))
                py_tokens = hgf_tkr.tokenize(line)
                @test jl_tokens == py_tokens
                jl_indices = collect(reinterpret(Int32, encode(tkr, line).input.tok))
                py_indices = collect(hgf_tkr(line)["input_ids"]) .+ 1
                @test jl_indices == py_indices

                py_input = torch.tensor(collect(hgf_tkr(line)["input_ids"])).reshape(1, length(py_indices))
                jl_input = encode(tkr, [line]).input.tok
                py_states = hgf_model(py_input)
                jl_states = model(jl_input)

                if haskey(jl_states, :start_logits)
                    py_result = rowmaj2colmaj(py_states["start_logits"].detach().numpy())
                    jl_result = jl_states.start_logits
                    py_result2 = rowmaj2colmaj(py_states["end_logits"].detach().numpy())
                    jl_result2 = jl_states.end_logits

                    jl_result = reshape(jl_result, Val(ndims(py_result)))
                    jl_result2 = reshape(jl_result2, Val(ndims(py_result2)))

                    jl_pred = Flux.onecold(jl_result)
                    py_pred = Flux.onecold(py_result)
                    jl_pred2 = Flux.onecold(jl_result2)
                    py_pred2 = Flux.onecold(py_result2)

                    @test jl_pred == py_pred
                    @test jl_pred2 == py_pred2

                    if jl_pred != py_pred || jl_pred2 != py_pred2
                        push!(error_samples, (line, (jl_input, jl_pred, jl_pred2), (py_input, py_pred, py_pred2)))
                    end
                else
                    if haskey(jl_states, :logits)
                        py_result = rowmaj2colmaj(py_states["logits"].detach().numpy())
                        jl_result = jl_states.logits
                    elseif haskey(jl_states, :prediction_logits)
                        py_result = rowmaj2colmaj(py_states["prediction_logits"].detach().numpy())
                        jl_result = jl_states.prediction_logits
                    end
                    jl_result = reshape(jl_result, Val(ndims(py_result)))

                    jl_pred = Flux.onecold(jl_result)
                    py_pred = Flux.onecold(py_result)
                    @test jl_pred == py_pred

                    if jl_pred != py_pred
                        push!(error_samples, (line, (jl_input, jl_pred), (py_input, py_pred)))
                    end
                end
            end
        catch e
            isnothing(fd) || close(fd)
            rethrow(e)
        end
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
