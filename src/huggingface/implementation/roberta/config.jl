@hgfcfg :roberta struct HGFRobertaConfig
    vocab_size::Int = 30522
    hidden_size::Int = 768
    num_hidden_layers::Int = 12
    num_attention_heads::Int = 12
    intermediate_size::Int = 3072
    hidden_act::String = "gelu"
    hidden_dropout_prob::Float64 = 0.1
    attention_probs_dropout_prob::Float64 = 0.1
    max_position_embeddings::Int = 512
    type_vocab_size::Int = 2
    initializer_range::Float32 = 0.02
    layer_norm_eps::Float32 = 1e-12
    pad_token_id::Int = 1
    bos_token_id::Int = 0
    eos_token_id::Int = 2
    position_embedding_type::String = "absolute"
    classifier_dropout::Nothing = nothing
end
