using ..TextEncoders: GPTTextEncoder, grouping_sentence
using FuncPipelines
using TextEncodeBase: SequenceTemplate, RepeatedTerm, InputTerm

tokenizer_type(T::Val{Symbol("openai-gpt")}) = T
function encoder_construct(::Val{Symbol("openai-gpt")}, tokenizer, vocab; process = nothing, kwargs...)
    if isnothing(process)
        # replacing the default process since huggingface openai-gpt doesn't have the special tokens
        # and being treated as a text generation model only
        process = Pipelines(
            Pipeline{:token}(grouping_sentence, :token),
            Pipeline{:token}(SequenceTemplate(RepeatedTerm(InputTerm{String}()))(Val(1)), :token),
        )
    end
    return GPTTextEncoder(tokenizer, vocab; process, kwargs...)
end

function extract_fast_tkr_kwargs(
    ::Val{Symbol("openai-gpt")}, config, special_tokens;
    model_max_length = get(config, :n_positions, 512), kw...
)
    kwargs = Dict{Symbol, Any}()
    kwargs[:trunc] = model_max_length
    return kwargs
end
