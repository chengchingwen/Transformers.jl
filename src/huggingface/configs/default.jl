macro defaultdef(ex)
    return var"@defaultdef"(__source__, __module__, nothing, ex)
end
macro defaultdef(model_type, ex)
    if model_type isa Symbol
        @assert model_type != :nothing "use `@defaultdef struct ... end` instead of `@defaultdef nothing struct ... end`"
        model_type = QuoteNode(model_type)
    end
    @assert model_type isa Union{QuoteNode, Nothing}
    @assert Meta.isexpr(ex, :struct, 3) "only accept struct definition like Base.@kwdef"
    mut, type, fields = ex.args
    @assert !mut "mutable default is not allowed"
    @assert type isa Symbol
    @assert Meta.isexpr(fields, :block)
    body = Expr[]
    values = []
    for field in fields.args
        if field isa Expr
            @assert Meta.isexpr(field, :(=), 2) "no default provided"
            name, default = field.args
            @assert Meta.isexpr(name, :(::), 2) "no type provided for entry $name"
            push!(body, esc(name))
            push!(values, esc(default))
        elseif field isa Symbol
            error("no default provided")
        end
    end
    push!(body, :($type() = new($(values...))))
    structdef = Expr(:struct, false, :($type <: AbstractHGFConfigDefault), Expr(:block, body...))
    if isnothing(model_type)
        return quote
            $structdef
        end
    else
        return quote
            $structdef
            $(@__MODULE__).getdefault(::HGFConfig{$model_type}) = $(esc(type))()
        end
    end
end

# a collection of default values
abstract type AbstractHGFConfigDefault end

@defaultdef struct HGFConfigDefault
    name_or_path::String = ""
    output_hidden_states::Bool = false
    output_attentions::Bool = false
    return_dict::Bool = true
    # encoder-decoder
    is_encoder_decoder::Bool = false
    is_decoder::Bool = false
    cross_attention_hidden_size::Nothing = nothing
    add_cross_attention::Bool = false
    tie_encoder_decoder::Bool = false
    # misc
    prune_heads::Tuple{} = ()
    chunk_size_feed_forward::Int = 0
    # sequence generation
    max_length::Int = 20
    min_length::Int = 10
    do_sample::Bool = false
    early_stopping::Bool = false
    num_beams::Int = 1
    num_beam_groups::Int = 1
    diversity_penalty::Float64 = 0.
    temeperature::Float64 = 1.
    top_k::Int = 50
    top_p::Float64 = 1.
    repetition_penalty::Float64 = 1.
    length_penalty::Float64 = 1.
    no_repeat_ngram_size::Int = 0
    encoder_no_repeat_ngram_size::Int = 0
    bad_words_ids::Nothing = nothing
    num_return_sequence::Int = 1
    output_scores::Bool = false
    return_dict_in_generate::Bool = false
    forced_bos_token_id::Nothing = nothing
    forced_eos_token_id::Nothing = nothing
    remove_invalid_values::Nothing = nothing
    # fine-tune task
    architectures::Nothing = nothing
    finetuning_task::Nothing = nothing
    id2label::Nothing = nothing
    label2id::Nothing = nothing
    num_labels::Int = 2
    task_specific_params::Nothing = nothing
    problem_type::Nothing = nothing
    # tokenizer
    tokenizer_class::Nothing = nothing
    prefix::Nothing = nothing
    bos_token_id::Nothing = nothing
    pad_token_id::Nothing = nothing
    eos_token_id::Nothing = nothing
    decoder_start_token_id::Nothing = nothing
    sep_token_id::Nothing = nothing
    # pytorch
    torchscript::Bool = false
    tie_word_embeddings::Bool = true
    torch_dtype::Nothing = nothing
    # tf
    use_bfloat16::Bool = false
    xla_device::Nothing = nothing
    transformers_version::Nothing = nothing
end

const DEFAULT_PRETRAIN_CONFIG = HGFConfigDefault()

Base.getproperty(cfg::AbstractHGFConfigDefault, sym::Symbol) = hasfield(typeof(cfg), sym) ? getfield(cfg, sym) : getfield(DEFAULT_PRETRAIN_CONFIG, sym)
