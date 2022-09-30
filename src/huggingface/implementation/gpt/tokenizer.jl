using ..Transformers.GenerativePreTrain
using FuncPipelines
using TextEncodeBase: SequenceTemplate, RepeatedTerm, InputTerm
using ..Basic: grouping_sentence

tokenizer_type(T::Val{Symbol("openai-gpt")}) = T
function encoder_construct(::Val{Symbol("openai-gpt")}, tokenizer, vocab; process = nothing, kwargs...)
    if isnothing(process)
        # replacing the default process since huggingface openai-gpt doesn't have the special tokens
        # and being treated as a text generation model only
        process = Pipelines(
            Pipeline{:tok}(grouping_sentence, :tok),
            Pipeline{:tok}(SequenceTemplate(RepeatedTerm(InputTerm{String}()))(Val(1)), :tok),
        )
    end
    return GPTTextEncoder(tokenizer, vocab; process, kwargs...)
end


function extract_tkr_kwargs(
    ::Val{Symbol("openai-gpt")}, config, special_tokens;
    unk_token = "<unk>", model_max_length = get_key(config, :n_positions, 512), kw...
)
    if !isnothing(special_tokens)
        unk_token = get(special_tokens, :unk_token, unk_token)
    end

    kwargs = Dict{Symbol, Any}()
    kwargs[:trunc] = model_max_length

    slow_tkr_kwargs = Dict{Symbol, Any}()
    slow_tkr_kwargs[:unk_token] = unk_token

    return kwargs, slow_tkr_kwargs
end
