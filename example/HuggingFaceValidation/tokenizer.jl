using TimerOutputs
using TextEncodeBase: getvalue, nestedcall

function test_tokenizer(name, corpus; output = nothing)
    global torch, hgf_trf, config, vocab_size, config, pyconfig, to
    @info "Validate $name tokenizer with corpus $corpus"
    @testset "Tokenizer" begin
        isfile(corpus) || error("corpus file $corpus do not exist.")

        @info "Loading tokenizer in Python"
        hgf_tkr = @tryrun begin
            @timeit to "pyload" hgf_trf.AutoTokenizer.from_pretrained(name, config = pyconfig)
        end "Failed to load the tokenizer in Python"
        @info "Python tokenizer loaded successfully"

        @info "Loading tokenizer in Julia"
        tkr = @tryrun begin
            @timeit to "jlload" HuggingFace.load_tokenizer(name; config)
        end "Failed to load the tokenizer in Julia"
        @info "Julia tokenizer loaded successfully"

        @info "Testing: Tokenizer"
        trunc = !isnothing(get(tkr.config, :trunc, nothing))
        fd = nothing
        out_fd = nothing
        prev_line = nothing
        try
            fd = open(corpus)
            isnothing(output) || (out_fd = open(output, "w+"))
            for line in eachline(fd)
                # single input
                isempty(line) && continue
                jl_tokens = @timeit to "jltokenize" TextEncodeBase.getvalue.(TextEncodeBase.tokenize(tkr, line))
                py_tokens = @timeit to "pytokenize" hgf_tkr.tokenize(line; truncation = false)
                @test jl_tokens == py_tokens
                jl_indices = collect(reinterpret(Int32, encode(tkr, line).token))
                _py_indices = hgf_tkr(line; truncation = trunc)["input_ids"]
                py_indices = collect(_py_indices) .+ 1
                @test jl_indices == py_indices
                jl_decoded = @timeit to "jldecode" TextEncodeBase.decode_text(tkr, jl_indices)
                py_decoded = @timeit to "pydecode" hgf_tkr.decode(_py_indices)
                @test jl_decoded == py_decoded
                single_pass = jl_tokens == py_tokens && jl_decoded == py_decoded
                if !single_pass
                    println("Failed: ", repr(line))
                    isnothing(out_fd) || println(out_fd, line)
                end
                # pair input
                if !isnothing(prev_line) && single_pass
                    pair_jl_tokens =
                        @timeit to "jltokenize2" vcat(
                            nestedcall(getvalue, TextEncodeBase.tokenize(tkr, [[prev_line, line]]))[]...)
                    pair_py_tokens = @timeit to "pytokenize2" hgf_tkr.tokenize(prev_line, line; truncation = false)
                    @test pair_jl_tokens == pair_py_tokens
                    pair_jl_indices = reshape(
                        collect(reinterpret(Int32, encode(tkr, [[prev_line, line]]).token)), :)
                    pair_py_indices = collect(hgf_tkr(prev_line, line; truncation = trunc)["input_ids"]) .+ 1
                    pair_pass = pair_jl_indices == pair_py_indices
                    @test pair_jl_indices == pair_py_indices
                    if !pair_pass
                        println("Pair Failed: ", (prev_line, line))
                    end
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
