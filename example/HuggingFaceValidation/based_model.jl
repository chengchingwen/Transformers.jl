using TimerOutputs

function test_based_model(name, n; max_error = 1e-2, mean_error = 1e-4)
    global torch, hgf_trf, vocab_size, config, pyconfig, to
    @info "Validate $name based model"
    @testset "Based Model" begin
        @info "Loading based model in Python"
        global hgf_model = @tryrun begin
            @timeit to "pyload" hgf_trf.AutoModel.from_pretrained(name, config = pyconfig)
        end "Failed to load the model in Python"
        @info "Python model loaded successfully"

        @info "Loading based model in Julia"
        global model = @tryrun begin
            @timeit to "jlload" HuggingFace.load_model(name; config = config)
        end "Failed to load the model in Julia"
        @info "Julia model loaded successfully"

        @info "Testing: Based model forward"
        for i = 1:n
            if model isa HuggingFace.HGFCLIPModel
                eos_id = config.text_config.eos_token_id
                len = config.text_config.max_position_embeddings
                indices = rand(1:vocab_size-1, len)
                indices[len - 3] = eos_id == 2 ? vocab_size : eos_id + 1
                pyindices = torch.tensor(indices .- 1).reshape(1, len)
                image_size = config.vision_config.image_size
                pixels = randn(Float32, image_size, image_size, 3, 1)
                pypixels = torch.tensor(permutedims(pixels, 4:-1:1))
                py_results = @timeit to "pyforward" hgf_model(input_ids = pyindices, pixel_values = pypixels)
                py_text_result1 = rowmaj2colmaj(py_results["text_model_output"]["last_hidden_state"].detach().numpy())
                py_text_result2 = rowmaj2colmaj(py_results["text_embeds"].detach().numpy())
                py_text_result3 = rowmaj2colmaj(py_results["logits_per_text"].detach().numpy())
                py_vision_result1 = rowmaj2colmaj(py_results["vision_model_output"]["last_hidden_state"].detach().numpy())
                py_vision_result2 = rowmaj2colmaj(py_results["image_embeds"].detach().numpy())
                py_vision_result3 = rowmaj2colmaj(py_results["logits_per_image"].detach().numpy())
                jl_results = @timeit to "jlforward" model((
                    text_input = (token = reshape(indices, len, 1), sequence_mask = Masks.LengthMask([len - 3])),
                    vision_input = (pixel = pixels,)))
                jl_text_result1 = jl_results.text_output.hidden_state
                jl_text_result2 = jl_results.embeddings.text
                jl_text_result3 = jl_results.logits.text
                jl_vision_result1 = jl_results.vision_output.hidden_state
                jl_vision_result2 = jl_results.embeddings.vision
                jl_vision_result3 = jl_results.logits.vision
                textdiff1 = (py_text_result1 .- jl_text_result1) .^ 2
                textdiff2 = (py_text_result2 .- jl_text_result2) .^ 2
                textdiff3 = (py_text_result3 .- jl_text_result3) .^ 2
                visiondiff1 = (py_vision_result1 .- jl_vision_result1) .^ 2
                visiondiff2 = (py_vision_result2 .- jl_vision_result2) .^ 2
                visiondiff3 = (py_vision_result3 .- jl_vision_result3) .^ 2
                @debug "text-diff" mean1 = mean(textdiff1) max1 = maximum(textdiff1) mean2 = mean(textdiff2) max2 = maximum(textdiff2) mean3 = mean(textdiff3) max3 = maximum(textdiff3)
                @debug "vision-diff" mean1 = mean(visiondiff1) max1 = maximum(visiondiff1) mean2 = mean(visiondiff2) max2 = maximum(visiondiff2) mean3 = mean(visiondiff3) max3 = maximum(visiondiff3)
                @test maximum(textdiff1) < max_error
                @test mean(textdiff1) < mean_error
                @test maximum(textdiff2) < max_error
                @test mean(textdiff2) < mean_error
                @test maximum(textdiff3) < max_error
                @test mean(textdiff3) < mean_error
                @test maximum(visiondiff1) < max_error
                @test mean(visiondiff1) < mean_error
                @test maximum(visiondiff2) < max_error
                @test mean(visiondiff2) < mean_error
                @test maximum(visiondiff3) < max_error
                @test mean(visiondiff3) < mean_error
            elseif HuggingFace.is_seq2seq(model)
                len1 = rand(50:100)
                len2 = rand(50:100)
                indices1 = rand(1:vocab_size, len1)
                indices2 = rand(1:vocab_size, len2)
                pyindices1 = torch.tensor(indices1 .- 1).reshape(1, len1)
                pyindices2 = torch.tensor(indices2 .- 1).reshape(1, len2)
                py_results = @timeit to "pyforward" hgf_model(input_ids=pyindices1, decoder_input_ids=pyindices2)
                py_result1 = rowmaj2colmaj(py_results["last_hidden_state"].detach().numpy())
                py_result2 = rowmaj2colmaj(py_results["encoder_last_hidden_state"].detach().numpy())
                jl_results = @timeit to "jlforward" model((
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
                py_results = @timeit to "pyforward" hgf_model(pyindices)
                py_result = rowmaj2colmaj(py_results["last_hidden_state"].detach().numpy())
                jl_results = @timeit to "jlforward" model((token = reshape(indices, len, 1),))
                jl_result = jl_results.hidden_state
                diff = (py_result .- jl_result) .^ 2
                @debug "diff" mean = mean(diff) max = maximum(diff)
                @test maximum(diff) < max_error
                @test mean(diff) < mean_error
            end
        end
    end
end
