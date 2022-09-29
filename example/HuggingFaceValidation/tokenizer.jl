function test_tokenizer(name, corpus)
    global torch, hgf_trf, config, vocab_size
    @info "Validate $name tokenizer with corpus $corpus"
    @testset "Tokenizer" begin
        isfile(corpus) || error("corpus file $corpus do not exist.")

        @info "Loading tokenizer in Python"
        hgf_tkr = @tryrun begin
            hgf_trf.AutoTokenizer.from_pretrained(name)
        end "Failed to load the tokenizer in Python"
        @info "Python tokenizer loaded successfully"

        @info "Loading tokenizer in Julia"
        tkr = @tryrun begin
            HuggingFace.load_hgf_pretrained("$name:tokenizer")
        end "Failed to load the tokenizer in Julia"
        @info "Julia tokenizer loaded successfully"

        @info "Testing: Tokenizer"
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
            end
        catch e
            isnothing(fd) || close(fd)
            rethrow(e)
        end
    end
end
