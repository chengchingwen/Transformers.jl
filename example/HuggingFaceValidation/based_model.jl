function test_based_model(name, n; max_error = 1e-2, mean_error = 1e-4)
    global torch, hgf_trf, vocab_size
    @info "Validate $name based model"
    @testset "Based Model" begin
        @info "Loading based model in Python"
        hgf_model = @tryrun begin
            hgf_trf.AutoModel.from_pretrained(name)
        end "Failed to load the model in Python"
        @info "Python model loaded successfully"

        @info "Loading based model in Julia"
        model = @tryrun begin
            HuggingFace.load_model(name)
        end "Failed to load the model in Julia"
        @info "Julia model loaded successfully"

        @info "Testing: Based model forward"
        for i = 1:n
            if HuggingFace.is_seq2seq(model)
                len1 = rand(50:100)
                len2 = rand(50:100)
                indices1 = rand(1:vocab_size, len1)
                indices2 = rand(1:vocab_size, len2)
                pyindices1 = torch.tensor(indices1 .- 1).reshape(1, len1)
                pyindices2 = torch.tensor(indices2 .- 1).reshape(1, len2)
                py_results = hgf_model(input_ids=pyindices1, decoder_input_ids=pyindices2)
                py_result1 = rowmaj2colmaj(py_results["last_hidden_state"].detach().numpy())
                py_result2 = rowmaj2colmaj(py_results["encoder_last_hidden_state"].detach().numpy())
                jl_results = model((
                    encoder_input = (token = reshape(indices1, len1, 1),),
                    decoder_input = (token = reshape(indices2, len2, 1),)))
                jl_result1 = jl_results.decoder_output.hidden_state
                jl_result2 = jl_results.encoder_output.hidden_state
                diff1 = (py_result1 .- jl_result1) .^ 2
                diff2 = (py_result2 .- jl_result2) .^ 2
                @debug "diff" mean1 = mean(diff1) max1 = maximum(diff1) mean2 = mean(diff2) max2 = maximum(diff2)
                @test maximum(diff1) < max_error
                @test mean(diff1) < mean_error
                @test maximum(diff2) < max_error
                @test mean(diff2) < mean_error
            else
                len = rand(50:100)
                indices = rand(1:vocab_size, len)
                pyindices = torch.tensor(indices .- 1).reshape(1, len)
                py_result = rowmaj2colmaj(hgf_model(pyindices)["last_hidden_state"].detach().numpy())
                jl_result = model((token = reshape(indices, len, 1),)).hidden_state
                diff = (py_result .- jl_result) .^ 2
                @debug "diff" mean = mean(diff) max = maximum(diff)
                @test maximum(diff) < max_error
                @test mean(diff) < mean_error
            end
        end
    end
end
