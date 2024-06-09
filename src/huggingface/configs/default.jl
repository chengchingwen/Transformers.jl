function hgfcfgm(model_type, ex)
    if model_type isa Symbol
        model_type != :nothing || error("syntax: use `@hgfcfg struct ... end` instead of `@hgfcfg nothing struct ... end`")
        model_type = QuoteNode(model_type)
    end
    model_type isa Union{QuoteNode, Nothing} || error("syntax: invalid model_type: $model_type")
    Meta.isexpr(ex, :struct, 3) || error("syntax: invalid definition")
    mut, type, fields = ex.args
    mut && error("syntax: mutable is not allowed")
    (type isa Symbol && Meta.isexpr(fields, :block)) || error("syntax: invalid struct definition")
    expr = quote
        const $(esc(type)) = HGFConfig{$model_type}
    end
    values = Expr(:tuple)
    cur_loc = nothing
    for field in fields.args
        field isa Symbol && error("syntax: no default provided: $field")
        if field isa LineNumberNode
            cur_loc = field
            continue
        end
        Meta.isexpr(field, :(=), 2) || error("syntax: invalid field around $cur_loc: $field")
        entry, value = field.args
        entry isa Symbol && error("syntax: no type provided for field `$entry`")
        Meta.isexpr(entry, :(::), 2) || error("syntax: invalid field entry: $entry")
        name, ftype = entry.args
        push!(values.args, :($name = convert($ftype, $value)))
    end
    alias_tuple = alias_tuplem(values)
    push!(expr.args, :($(@__MODULE__).getnamemap(::Type{<:HGFConfig{$model_type}}) = $(alias_tuple.args[2])))
    alias_tuple.args[2] = :(getnamemap(HGFConfig{$model_type}))
    push!(expr.args, :($(@__MODULE__).getdefault(::Type{<:HGFConfig{$model_type}}) = $alias_tuple))
    return expr
end

macro hgfcfg(model_type, ex)
    return hgfcfgm(model_type, ex)
end

const DEFAULT_PRETRAIN_CONFIG = @alias (
    name_or_path = "",
    output_hidden_states = false,
    output_attentions = false,
    return_dict = true,
    # encoder-decoder
    is_encoder_decoder = false,
    is_decoder = false,
    cross_attention_hidden_size = nothing,
    add_cross_attention = false,
    tie_encoder_decoder = false,
    # misc
    prune_heads = (),
    chunk_size_feed_forward = 0,
    # sequence generation
    max_length = 20,
    min_length = 10,
    do_sample = false,
    early_stopping = false,
    num_beams = 1,
    num_beam_groups = 1,
    diversity_penalty = 0.,
    temeperature = 1.,
    top_k = 50,
    top_p = 1.,
    repetition_penalty = 1.,
    length_penalty = 1.,
    no_repeat_ngram_size = 0,
    encoder_no_repeat_ngram_size = 0,
    bad_words_ids = nothing,
    num_return_sequence = 1,
    output_scores = false,
    return_dict_in_generate = false,
    forced_bos_token_id = nothing,
    forced_eos_token_id = nothing,
    remove_invalid_values = nothing,
    # fine-tune task
    architectures = nothing,
    finetuning_task = nothing,
    id2label = nothing,
    label2id = nothing,
    num_labels = 2,
    task_specific_params = nothing,
    problem_type = nothing,
    # tokenizer
    tokenizer_class = nothing,
    prefix = nothing,
    bos_token_id = nothing,
    pad_token_id = nothing,
    eos_token_id = nothing,
    decoder_start_token_id = nothing,
    sep_token_id = nothing,
    clean_up_tokenization_spaces = true,
    split_special_tokens = false,
    # pytorch
    torchscript = false,
    tie_word_embeddings = true,
    torch_dtype = nothing,
    # tf
    use_bfloat16 = false,
    xla_device = nothing,
    transformers_version = nothing,
)
