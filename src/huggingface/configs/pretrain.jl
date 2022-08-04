# a collection of default values
struct HGFPretrainedConfigBase <: HGFPretrainedConfig
    name_or_path::String
    output_hidden_states::Bool
    output_attentions::Bool
    return_dict::Bool
    # encoder-decoder
    is_encoder_decoder::Bool
    is_decoder::Bool
    cross_attention_hidden_size::Nothing
    add_cross_attention::Bool
    tie_encoder_decoder::Bool
    # misc
    prune_heads::Tuple{}
    chunk_size_feed_forward::Int
    # sequence generation
    max_length::Int
    min_length::Int
    do_sample::Bool
    early_stopping::Bool
    num_beams::Int
    num_beam_groups::Int
    diversity_penalty::Float64
    temeperature::Float64
    top_k::Int
    top_p::Float64
    repetition_penalty::Float64
    length_penalty::Float64
    no_repeat_ngram_size::Int
    encoder_no_repeat_ngram_size::Int
    bad_words_ids::Nothing
    num_return_sequence::Int
    output_scores::Bool
    return_dict_in_generate::Bool
    forced_bos_token_id::Nothing
    forced_eos_token_id::Nothing
    remove_invalid_values::Nothing
    # fine-tune task
    architectures::Nothing
    finetuning_task::Nothing
    id2label::Nothing
    label2id::Nothing
    num_labels::Int
    task_specific_params::Nothing
    problem_type::Nothing
    # tokenizer
    tokenizer_class::Nothing
    prefix::Nothing
    bos_token_id::Nothing
    pad_token_id::Nothing
    eos_token_id::Nothing
    decoder_start_token_id::Nothing
    sep_token_id::Nothing
    # pytorch
    torchscript::Bool
    tie_word_embeddings::Bool
    torch_dtype::Nothing
    # tf
    use_bfloat16::Bool
    xla_device::Nothing
    transformers_version::Nothing

    HGFPretrainedConfigBase() = new(
        "", false, false, true,
        false, false, nothing, false, false, (), 0,
        20, 10, false, false, 1, 1, 0.0, 1.0, 50, 1.0, 1.0, 1.0, 0, 0, nothing, 1,
        false, false, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing, 2, nothing, nothing,
        nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        false, true, nothing, false, nothing,
        nothing,
    )
end

struct UpdatedConfig{V} <: HGFPretrainedConfig
    config::HGFPretrainedConfig
    name::Symbol
    value::V
end

update(cfg::HGFPretrainedConfigBase, name::Symbol, v) = UpdatedConfig(cfg, name, v)
function update(cfg::UpdatedConfig, name::Symbol, v)
    if _name(cfg) == name
        return UpdatedConfig(_config(cfg), name, v)
    else
        return UpdatedConfig(
            update(_config(cfg), name, v),
            _name(cfg), _value(cfg))
    end
end

_name(cfg::UpdatedConfig) = getfield(cfg, :name)
_value(cfg::UpdatedConfig) = getfield(cfg, :value)
_config(cfg::UpdatedConfig) = getfield(cfg, :config)
_base(cfg::UpdatedConfig) = (_base ∘ _config)(cfg)
_base(cfg::HGFPretrainedConfigBase) = cfg

Base.get(cfg::HGFPretrainedConfigBase, k::Symbol, d) = hasfield(typeof(cfg), k) ? getfield(cfg, k) : d
Base.get(cfg::UpdatedConfig, k::Symbol, d) = k == _name(cfg) ? _value(cfg) : get(_config(cfg), k, d)
Base.getproperty(cfg::UpdatedConfig, k::Symbol) = k == _name(cfg) ? _value(cfg) : Base.getproperty(_config(cfg), k)

_get_cfg(cfg::UpdatedConfig, k) = k == _name(cfg) ? cfg : _get_cfg(_config(cfg), k)

_current_pair(cfg::UpdatedConfig) = _name(cfg)=>_value(cfg)

_updated(cfg::UpdatedConfig) = tuple(_current_pair(cfg), _updated(_config(cfg))...)
_updated(cfg::HGFPretrainedConfigBase) = ()

Base.iterate(cfg::UpdatedConfig) = iterate(cfg, _name(cfg))
function Base.iterate(cfg::UpdatedConfig, i)
    isnothing(i) && return nothing
    cur = _get_cfg(cfg, i)
    p = _current_pair(cur)
    nc = _config(cur)
    if nc isa UpdatedConfig
        return p, _name(nc)
    else
        return p, nothing
    end
end

function pretrained_config(; kws...)
    global PRETRAIN_CONFIG
    p = PRETRAIN_CONFIG
    for (k, v) in kws
        if k == :id2label && !isnothing(v)
            v = Dict{Int, String}(parse(Int, key)=>val for (key, val) in v)
        elseif k == :label2id && !isnothing(v)
            v = convert(Dict{String, Int}, v)
        end
        p = update(p, k, v)
    end
    return p
end

Base.length(cfg::UpdatedConfig) = length(_updated(cfg))
function Base.summary(io::IO, x::UpdatedConfig)
    n = length(x)
    print(io, typeof(x))
    print(io, " with ", n, (n==1 ? " entry" : " entries"))
end

const PRETRAIN_CONFIG = HGFPretrainedConfigBase()
