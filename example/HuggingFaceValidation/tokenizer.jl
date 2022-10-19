using TextEncodeBase: getvalue, nestedcall

function test_tokenizer(name, corpus; output = nothing)
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
        out_fd = nothing
        prev_line = nothing
        try
            fd = open(corpus)
            isnothing(output) || (out_fd = open(output, "w+"))
            for line in eachline(fd)
                # single input
                isempty(line) && continue
                jl_tokens = TextEncodeBase.getvalue.(TextEncodeBase.tokenize(tkr, line))
                py_tokens = hgf_tkr.tokenize(line)
                @test jl_tokens == py_tokens
                jl_indices = collect(reinterpret(Int32, encode(tkr, line).input.tok))
                py_indices = collect(hgf_tkr(line)["input_ids"]) .+ 1
                @test jl_indices == py_indices

                single_pass = jl_tokens == py_tokens
                if !single_pass
                    println("Failed: ", repr(line))
                    isnothing(out_fd) || println(out_fd, line)
                end
                # pair input
                if !isnothing(prev_line) && single_pass
                    pair_jl_tokens =
                        vcat(nestedcall(getvalue, TextEncodeBase.tokenize(tkr, [[prev_line, line]]))[]...)
                    pair_py_tokens = hgf_tkr.tokenize(prev_line, line)
                    @test pair_jl_tokens == pair_py_tokens
                    pair_jl_indices = reshape(
                        collect(reinterpret(Int32, encode(tkr, [[prev_line, line]]).input.tok)), :)
                    pair_py_indices = collect(hgf_tkr(prev_line, line)["input_ids"]) .+ 1
                    @test pair_jl_indices == pair_py_indices
                end
                single_pass && (prev_line = line)
            end
        catch e
            isnothing(fd) || close(fd)
            isnothing(out_fd) || close(out_fd)
            rethrow(e)
        finally
            isnothing(fd) || close(fd)
            isnothing(out_fd) || close(out_fd)
        end
    end
end
