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
            HuggingFace.load_hgf_pretrained("$name:model")
        end "Failed to load the model in Julia"
        @info "Julia model loaded successfully"

        @info "Testing: Based model forward"
        for i = 1:n
            len = rand(50:100)
            indices = rand(1:vocab_size, len)
            pyindices = torch.tensor(indices .- 1).reshape(1, len)
            py_result = rowmaj2colmaj(hgf_model(pyindices)["last_hidden_state"].detach().numpy())
            jl_result = model(reshape(indices, len, 1)).last_hidden_state
            diff = (py_result .- jl_result) .^ 2
            @test maximum(diff) < max_error
            @test mean(diff) < mean_error
        end
    end
end
