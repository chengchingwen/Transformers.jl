@assert !isnothing(HFHUB_Token)
using Artifacts, LazyArtifacts
const artifact_dir = @artifact_str("xnli_dev")
const xnli = joinpath(artifact_dir, "xnli-dev.txt")
using PythonCall
const hgf_trf = pyimport("transformers")
const hgf_cache_dir = mktempdir(; prefix = "JL_TRF_TEST_HGF_CAHCE_", cleanup = true)

using Transformers
using TimerOutputs
using TextEncodeBase
using TextEncodeBase: getvalue, nestedcall

macro tryrun(ex, msg = nothing)
    err_msg = isnothing(msg) ? nothing : :(@error $msg)
    return quote
        try
            $(esc(ex))
        catch e
            $err_msg
            rethrow(e)
        end
    end
end

function test_tokenizer(name, corpus; to = TimerOutput())
    global hgf_trf, HFHUB_Token
    @info "Validate $name tokenizer"
    @info "Load $name configure file in Julia"
    config = @tryrun begin
        @timeit to "jlload cfg" begin
            cfg = HuggingFace.load_config(name; auth_token = HFHUB_Token, cache = false)
            HuggingFace.HGFConfig(cfg; layer_norm_eps = 1e-9, layer_norm_epsilon = 1e-9)
        end
    end "Failed to load $name configure file in Julia, probably unsupported"
    @info "Load $name configure file in Python"
    pyconfig = @tryrun begin
        @timeit to "pyload cfg" hgf_trf.AutoConfig.from_pretrained(name, token = HFHUB_Token, force_download = true,
                                                                   cache_dir = hgf_cache_dir,
                                                                   layer_norm_eps = 1e-9, layer_norm_epsilon = 1e-9)
    end "Failed to load $name configure file in Python, probably unsupported"
    vocab_size = if haskey(config, :vocab_size)
        config.vocab_size
    else
        @warn "Configure file doesn't have vocab_size information. Use 10000 as a default value"
        10000
    end
    @info "Loading $name tokenizer in Python"
    hgf_tkr = @tryrun begin
        @timeit to "pyload tkr" hgf_trf.AutoTokenizer.from_pretrained(name, config = pyconfig, token = HFHUB_Token,
                                                                      force_download = true,
                                                                      cache_dir = hgf_cache_dir)
    end "Failed to load $name tokenizer in Python"
    @info "Python $name tokenizer loaded successfully"
    @info "Loading $name tokenizer in Julia"
    tkr = @tryrun begin
        @timeit to "jlload tkr" HuggingFace.load_tokenizer(name; config, auth_token = HFHUB_Token, cache = false)
    end "Failed to load $name tokenizer in Julia"
    @info "Julia $name tokenizer loaded successfully"
    @info "Testing: $name Tokenizer"
    trunc = !isnothing(get(tkr.config, :trunc, nothing))
    prev_line = nothing
    for line in corpus
        # single input
        isempty(line) && continue
        jl_tokens = @timeit to "jltokenize" TextEncodeBase.getvalue.(TextEncodeBase.tokenize(tkr, line))
        _py_tokens = @timeit to "pytokenize" hgf_tkr.tokenize(line; truncation = false)
        py_tokens = pyconvert(Vector{String}, _py_tokens)
        @test jl_tokens == py_tokens
        jl_indices = collect(reinterpret(Int32, encode(tkr, line).token))
        _py_indices = hgf_tkr(line; truncation = trunc)["input_ids"]
        py_indices = pyconvert(Vector{Int}, _py_indices) .+ 1
        @test jl_indices == py_indices
        jl_decoded = @timeit to "jldecode" TextEncodeBase.decode_text(tkr, jl_indices)
        _py_decoded = @timeit to "pydecode" hgf_tkr.decode(_py_indices)
        py_decoded = pyconvert(String, _py_decoded)
        @test jl_decoded == py_decoded

        single_pass = jl_tokens == py_tokens && jl_indices == py_indices && jl_decoded == py_decoded
        if !single_pass
            println("Failed: ", repr(line))
        end
        # pair input
        if !isnothing(prev_line) && single_pass
            pair_jl_tokens =
                @timeit to "jltokenize2" vcat(
                    nestedcall(getvalue, TextEncodeBase.tokenize(tkr, [[prev_line, line]]))[]...)
            _pair_py_tokens = @timeit to "pytokenize2" hgf_tkr.tokenize(prev_line, line; truncation = false)
            pair_py_tokens = pyconvert(Vector{String}, _pair_py_tokens)
            @test pair_jl_tokens == pair_py_tokens
            pair_jl_indices = reshape(
                collect(reinterpret(Int32, encode(tkr, [[prev_line, line]]).token)), :)
            _pair_py_indices = hgf_tkr(prev_line, line; truncation = trunc)["input_ids"]
            pair_py_indices = pyconvert(Vector{Int}, _pair_py_indices) .+ 1
            @test pair_jl_indices == pair_py_indices
            pair_pass = pair_jl_tokens == pair_py_tokens && pair_jl_indices == pair_py_indices
            if !pair_pass
                println("Failed: ", (prev_line, line))
            end
        end
        single_pass && (prev_line = line)
    end
    return to
end

@testset "HuggingFace Tokenizer" begin
    corpus = readlines(xnli)
    for name in [
        "bert-base-cased", "bert-base-uncased", "roberta-base", "gpt2", "t5-small", "google/flan-t5-xl",
        "EleutherAI/pythia-70m", "databricks/dolly-v2-3b", "bigscience/bloom-560m", "meta-llama/Llama-2-7b-chat-hf",
        "facebook/bart-base", "openai/clip-vit-base-patch32"
    ]
        @testset "$name Tokenizer" begin
            to = TimerOutput()
            @timeit to "$name Tokenizer" test_tokenizer(name, corpus; to = to)
            show(to)
            println()
        end
    end
end
