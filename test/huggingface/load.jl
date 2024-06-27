@testset "Load" begin
    using Logging
    using Transformers.HuggingFace
    model_list = Dict([
        :bart => [:Model],
        :bert => :[
            Model, ForPreTraining, LMHeadModel, ForMaskedLM, ForNextSentencePrediction,
            ForSequenceClassification, ForTokenClassification, ForQuestionAnswering,
        ].args,
        :roberta => :[
            Model, ForMaskedLM, ForCausalLM, ForSequenceClassification, ForTokenClassification, ForQuestionAnswering,
        ].args,
        :gpt2 => [:Model, :LMHeadModel],
        :t5 => [:Model, :ForConditionalGeneration],
        :gpt_neo => [:Model, :ForCausalLM],
        :gptj => [:Model, :ForCausalLM],
        :gpt_neox => [:Model, :ForCausalLM],
        :bloom => [:Model, :ForCausalLM],
        :phi => [:Model, :ForCausalLM], 
        :clip => [:Model],
        # :llama => [:Model, :ForCausalLM], No hf-internal-testing/tiny-random-$hgf_type_name
    ])

    for (model_name, task_list) in model_list
        for task_type in task_list
            @info "Testing: $model_name - $task_type"
            GC.gc(true)
            model_type = HuggingFace.get_model_type(model_name, Symbol(lowercase(String(task_type))))
            hgf_type_name = chop(String(Base.typename(model_type).name), head = 3, tail = 0)
            hgf_model_name = "hf-internal-testing/tiny-random-$hgf_type_name"
            cfg = load_config(hgf_model_name; cache = false)
            model = nothing
            model = @test_logs min_level=Logging.Debug load_model(model_name, hgf_model_name, task_type;
                                                                  config = cfg, cache = false)
            @test model isa model_type
            isnothing(model) && continue
            state_dict1 = HuggingFace.get_state_dict(model)
            model2 = nothing
            model2 = @test_logs min_level=Logging.Debug load_model(model_type, cfg, state_dict1)
            isnothing(model2) && continue
            state_dict2 = HuggingFace.get_state_dict(model2)
            @test state_dict1 == state_dict2
            @test state_dict1 !== state_dict2

            if task_type != :Model
                if model isa Union{
                    HuggingFace.HGFBertLMHeadModel, HuggingFace.HGFBertForMaskedLM,
                    HuggingFace.HGFBertForTokenClassification, HuggingFace.HGFBertForQuestionAnswering,
                }
                    @test_logs (:debug, "bert.pooler.dense.weight not found, initialized.") (
                        :debug, "bert.pooler.dense.bias not found, initialized."
                    ) min_level=Logging.Debug load_model(model_name, hgf_model_name, :model, state_dict1;
                                                         config = cfg, cache = false)
                elseif model isa Union{
                    HuggingFace.HGFRobertaForCausalLM, HuggingFace.HGFRobertaForMaskedLM,
                    HuggingFace.HGFRobertaForSequenceClassification,
                    HuggingFace.HGFRobertaForTokenClassification, HuggingFace.HGFRobertaForQuestionAnswering,
                }
                    @test_logs (:debug, "roberta.pooler.dense.weight not found, initialized.") (
                        :debug, "roberta.pooler.dense.bias not found, initialized."
                    ) min_level=Logging.Debug load_model(model_name, hgf_model_name, :model, state_dict1;
                                                         config = cfg, cache = false)
                else
                    @test_logs min_level=Logging.Debug load_model(model_name, hgf_model_name, :model, state_dict1;
                                                                  config = cfg, cache = false)
                end
            end
        end
    end
end
