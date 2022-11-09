using Functors
using NeuralAttentionlib
using NeuralAttentionlib: rms_layer_norm, generic_multihead_qkv_attention, weighted_sum_mixing, $,
    normalized_score, masked_score, GenericMaskOp, dot_product_score,
    scalar_relative_position_embedding, t5_bucketed_position_id, t5_causal_bucketed_position_id,
    CausalMask

function FakeTHLinear(config::HGFT5Config, hi, ho, factor; bias=true)
    weight = randn(Float32, ho, hi) .* Float32(factor)
    if bias
        bias = zeros(Float32, ho)
    else
        bias = nothing
    end
    FakeTHLinear(weight, bias)
end

struct HGFT5LayerNorm{E, W<:AbstractArray{E}} <: THModule
    weight::W
    var_ϵ::E
end
@functor HGFT5LayerNorm (weight,)

(ln::HGFT5LayerNorm)(x) = rms_layer_norm(ln.var_ϵ, ln.weight, x)

HGFT5LayerNorm(config::HGFT5Config) = HGFT5LayerNorm(ones(Float32, config.d_model), Float32(config.layer_norm_epsilon))

struct HGFT5DenseActDense{F, Di, Do} <: THModule
    act::F
    wi::Di
    wo::Do
end
@functor HGFT5DenseActDense (wi, wo)

(dd::HGFT5DenseActDense)(x) = dd.wo(dd.act.(dd.wi(x)))

HGFT5DenseActDense(config::HGFT5Config) = HGFT5DenseActDense(
    ACT2FN[Symbol(config.dense_act_fn)],
    FakeTHLinear(config, config.d_model, config.d_ff, config.initializer_factor / sqrt(config.d_model); bias=false),
    FakeTHLinear(config, config.d_ff, config.d_model, config.initializer_factor / sqrt(config.d_ff); bias=false),
)

struct HGFT5DenseGatedActDense{F, Di0, Di1, Do} <: THModule
    act::F
    wi0::Di0
    wi1::Di1
    wo::Do
end
@functor HGFT5DenseGatedActDense (wi0, wi1, wo)

function (dd::HGFT5DenseGatedActDense)(x)
    hidden_gelu = dd.wi0(x)
    hidden_linear = dd.wi1(x)
    return dd.wo(hidden_gelu .* hidden_linear)
end

HGFT5DenseGatedActDense(config::HGFT5Config) = HGFT5DenseGatedActDense(
    ACT2FN[Symbol(config.dense_act_fn)],
    FakeTHLinear(config, config.d_model, config.d_ff, config.initializer_factor / sqrt(config.d_model); bias=false),
    FakeTHLinear(config, config.d_model, config.d_ff, config.initializer_factor / sqrt(config.d_model); bias=false),
    FakeTHLinear(config, config.d_ff, config.d_model, config.initializer_factor / sqrt(config.d_ff); bias=false),
)

struct HGFT5LayerFF{D, L<:HGFT5LayerNorm} <: THModule
    DenseReluDense::D
    layer_norm::L
end
@functor HGFT5LayerFF

(ff::HGFT5LayerFF)(x) = ff.DenseReluDense(ff.layer_norm(x)) + x

HGFT5LayerFF(config::HGFT5Config) = HGFT5LayerFF(
    config.is_gated_act ?
    HGFT5DenseGatedActDense(config) : HGFT5DenseActDense(config),
    HGFT5LayerNorm(config),
)

struct HGFT5Attention{Q, K, V, O, E} <: THModule
    is_decoder::Bool
    num_head::Int
    num_bucket::Int
    max_distance::Int
    q::Q
    k::K
    v::V
    o::O
    relative_attention_bias::E
end
function Functors.functor(::Type{<:HGFT5Attention}, a)
    if isnothing(a.relative_attention_bias)
        (q = a.q, k = a.k, v = a.v, o = a.o), y->HGFT5Attention(a.is_decoder, a.num_head, a.num_bucket, a.max_distance, y..., nothing)
    else
        (q = a.q, k = a.k, v = a.v, o = a.o, relative_attention_bias = a.relative_attention_bias), y->HGFT5Attention(a.is_decoder, a.num_head, a.num_bucket, a.max_distance, y...)
    end
end

function t5_attention(head, q, k, v, position_bias, mask, is_decoder, n_bucket, max_distance)
    generic_multihead_qkv_attention(
        weighted_sum_mixing,
        normalized_score(softmax) $
        masked_score(GenericMaskOp(), mask) $
        scalar_relative_position_embedding(
            is_decoder ?
              t5_causal_bucketed_position_id(n_bucket, max_distance) :
              t5_bucketed_position_id(n_bucket, max_distance),
            position_bias,
        ) $
        dot_product_score,
        head, q, k, v
    )
end

function t5_attention(head, q, k, v, position_bias::Nothing, mask, is_decoder, n_bucket, max_distance)
    generic_multihead_qkv_attention(
        weighted_sum_mixing,
        normalized_score(softmax) $
        masked_score(GenericMaskOp(), mask) $
        dot_product_score,
        head, q, k, v
    )
end

function (a::HGFT5Attention)(q, k, position_bias = nothing; mask = nothing)
    if has_relative_attention_bias(a)
        relative_position_bias = isnothing(position_bias) ? a.relative_attention_bias.weight : position_bias
    else
        relative_position_bias = position_bias
    end
    y = t5_attention(a.num_head, a.q(q), a.k(k), a.v(k),
                     relative_position_bias, mask, a.is_decoder, a.num_bucket, a.max_distance)
    o = a.o(y)
    return o, relative_position_bias
end

has_relative_attention_bias(a::HGFT5Attention) = !isnothing(a.relative_attention_bias)

function FakeTHEmbedding(::Type{HGFT5Attention}, config, num, dims, factor)
    weight = randn(Float32, dims, num) .* factor
    FakeTHEmbedding(0, weight)
end

function HGFT5Attention(config::HGFT5Config; has_relative_attention_bias = false)
    factor = Float32(config.initializer_factor)
    return HGFT5Attention(
        config.is_decoder,
        config.num_heads,
        config.relative_attention_num_buckets,
        config.relative_attention_max_distance,
        FakeTHLinear(config, config.d_model, config.num_heads * config.d_kv, factor / sqrt(config.d_model * config.d_kv); bias=false),
        FakeTHLinear(config, config.d_model, config.num_heads * config.d_kv, factor / sqrt(config.d_model); bias=false),
        FakeTHLinear(config, config.d_model, config.num_heads * config.d_kv, factor / sqrt(config.d_model); bias=false),
        FakeTHLinear(config, config.d_model, config.num_heads * config.d_kv, factor / sqrt(config.num_heads * config.d_kv); bias=false),
        has_relative_attention_bias ?
        FakeTHEmbedding(HGFT5Attention, config, config.relative_attention_num_buckets,
                        config.num_heads, factor / sqrt(config.d_model)) : nothing
    )
end

struct HGFT5LayerSelfAttention{A, L} <: THModule
    SelfAttention::A
    layer_norm::L
end
@functor HGFT5LayerSelfAttention

function (sa::HGFT5LayerSelfAttention)(x, position_bias = nothing; mask = nothing)
    hidden_state = sa.layer_norm(x)
    a, relative_position_bias = sa.SelfAttention(hidden_state, hidden_state, position_bias; mask)
    return (a + x, relative_position_bias)
end

HGFT5LayerSelfAttention(config::HGFT5Config; has_relative_attention_bias = false) = HGFT5LayerSelfAttention(
    HGFT5Attention(config; has_relative_attention_bias),
    HGFT5LayerNorm(config),
)

struct HGFT5LayerCrossAttention{A, L} <: THModule
    EncDecAttention::A
    layer_norm::L
end
@functor HGFT5LayerCrossAttention

function (ca::HGFT5LayerCrossAttention)(x, k, position_bias = nothing; mask = nothing)
    a, relative_position_bias = ca.EncDecAttention(ca.layer_norm(x), k, position_bias; mask)
    return (a + x, relative_position_bias)
end

HGFT5LayerCrossAttention(config::HGFT5Config; has_relative_attention_bias = false) = HGFT5LayerCrossAttention(
    HGFT5Attention(config; has_relative_attention_bias),
    HGFT5LayerNorm(config),
)

struct HGFT5Block{T<:FakeTHModuleList} <: THModule
    layer::T
end
@functor HGFT5Block

function (b::HGFT5Block{<:FakeTHModuleList{2}})(x, position_bias = nothing; mask = nothing)
    a, bias = b.layer[1](x, position_bias; mask)
    return (hidden_state = b.layer[2](a), relative_position_bias = bias)
end

function (b::HGFT5Block{<:FakeTHModuleList{3}})(x, encoder_hidden_state,
                                                position_bias = nothing, cross_position_bias = nothing;
                                                mask = nothing, cross_mask = nothing)
    a, bias = b.layer[1](x, position_bias; mask = mask & CausalMask())
    y, cross_bias = b.layer[2](a, encoder_hidden_state, cross_position_bias; mask = cross_mask)
    return (hidden_state = b.layer[3](y), relative_position_bias = bias, cross_relative_position_bias = cross_bias)
end

HGFT5Block(config::HGFT5Config; has_relative_attention_bias = false) = HGFT5Block(FakeTHModuleList(
    HGFT5LayerSelfAttention(config; has_relative_attention_bias),
    (config.is_decoder ? (HGFT5LayerCrossAttention(config),) : ())...,
    HGFT5LayerFF(config),
))

# t5 model

abstract type HGFT5PreTrainedModel <: HGFPreTrainedModel end

struct HGFT5Stack{L<:FakeTHModuleList, E, LN} <: HGFT5PreTrainedModel
    embed_tokens::E
    block::L
    final_layer_norm::LN
end
@functor HGFT5Stack

function (s::HGFT5Stack{<:FakeTHModuleList{N, <:NTuple{N, HGFT5Block{<:FakeTHModuleList{2}}}}})(
    x, position_bias = nothing;
    mask = nothing
) where N
    embed = s.embed_tokens(x)
    hidden_state = embed
    for b in s.block
        output = b(hidden_state, position_bias; mask)
        hidden_state = output.hidden_state
        position_bias = output.relative_position_bias
    end
    return s.final_layer_norm(hidden_state)
end

function (s::HGFT5Stack{<:FakeTHModuleList{N, <:NTuple{N, HGFT5Block{<:FakeTHModuleList{3}}}}})(
    x, encoder_hidden_state, position_bias = nothing, cross_position_bias = nothing;
    mask = nothing, cross_mask = nothing
) where N
    embed = s.embed_tokens(x)
    hidden_state = embed
    for b in s.block
        output = b(hidden_state, encoder_hidden_state, position_bias, cross_position_bias; mask, cross_mask)
        hidden_state = output.hidden_state
        position_bias = output.relative_position_bias
        cross_position_bias = output.cross_relative_position_bias
    end
    return s.final_layer_norm(hidden_state)
end

function HGFT5Stack(config::HGFT5Config, embed_tokens)
    block = FakeTHModuleList(
        [HGFT5Block(config; has_relative_attention_bias = isone(i)) for i in 1:config.num_layers]
    )
    HGFT5Stack(embed_tokens, block, HGFT5LayerNorm(config))
end

struct HGFT5Model{E, ENC, DEC} <: HGFT5PreTrainedModel
    shared::E
    encoder::ENC
    decoder::DEC
end
@functor HGFT5Model

function (m::HGFT5Model)(input, decoder_input; mask = nothing, decoder_mask = nothing)
    encoder_output = m.encoder(input; mask)
    decoder_output = m.decoder(decoder_input, encoder_output; mask = decoder_mask)
    return (encoder_output = encoder_output, decoder_output = decoder_output)
end

function FakeTHEmbedding(::Type{HGFT5Model}, config, num, dims)
    factor = Float32(config.initializer_factor)
    weight = randn(Float32, dims, num) .* factor
    return FakeTHEmbedding(0, weight)
end

function HGFT5Model(config::HGFT5Config)
    shared = FakeTHEmbedding(HGFT5Model, config, config.vocab_size, config.d_model)
    enc_config = HGFT5Config(; merge(merge((;), config), (is_decoder = false, is_encoder_decoder = false))...)
    dec_config = HGFT5Config(; merge(merge((;), config), (is_decoder = true, is_encoder_decoder = false, num_layers = config.num_decoder_layers))...)
    encoder = HGFT5Stack(enc_config, shared)
    decoder = HGFT5Stack(dec_config, shared)
    return HGFT5Model(shared, encoder, decoder)
end

struct HGFT5ForConditionalGeneration{E, ENC, DEC, D} <: HGFT5PreTrainedModel
    shared::E
    encoder::ENC
    decoder::DEC
    lm_head::D
end
@functor HGFT5ForConditionalGeneration

function (m::HGFT5ForConditionalGeneration)(input, decoder_input; mask = nothing, decoder_mask = nothing)
    encoder_output = m.encoder(input; mask)
    decoder_output = m.decoder(decoder_input, encoder_output; mask = decoder_mask)
    if m.shared.weight === m.lm_head.weight'
        sequence_output = decoder_output .* (inv(sqrt(size(decoder_output, 1))))
    else
        sequence_output = decoder_output
    end
    lm_logits = m.lm_head(sequence_output)
    return (logits = lm_logits, encoder_output = encoder_output, decoder_output = decoder_output)
end

function (m::HGFT5ForConditionalGeneration)(input, decoder_input, labels; mask = nothing, decoder_mask = nothing)
    outputs = m(input, decoder_input; mask, decoder_mask)
    loss = Flux.logitcrossentropy(outputs.logits, labels)
    return merge((loss = loss,), outputs)
end

function HGFT5ForConditionalGeneration(config::HGFT5Config)
    shared = FakeTHEmbedding(HGFT5Model, config, config.vocab_size, config.d_model)
    enc_config = HGFT5Config(; merge(merge((;), config), (is_decoder = false, is_encoder_decoder = false))...)
    dec_config = HGFT5Config(; merge(merge((;), config), (is_decoder = true, is_encoder_decoder = false, num_layers = config.num_decoder_layers))...)
    encoder = HGFT5Stack(enc_config, shared)
    decoder = HGFT5Stack(dec_config, shared)
    lm_head_weight = config.tie_word_embeddings ?
        shared.weight' :
        randn(Float32, config.d_model, config.vocab_size) .* config.initializer_factor
    lm_head = FakeTHLinear(lm_head_weight)
    return HGFT5ForConditionalGeneration(shared, encoder, decoder, lm_head)
end

struct HGFT5EncoderModel{E, ENC} <: HGFT5PreTrainedModel
    shared::E
    encoder::ENC
end
@functor HGFT5EncoderModel

function (m::HGFT5EncoderModel)(input; mask = nothing)
    encoder_output = m.encoder(input; mask)
    return encoder_output
end

function HGFT5EncoderModel(config::HGFT5Config)
    shared = FakeTHEmbedding(HGFT5Model, config, config.vocab_size, config.d_model)
    enc_config = HGFT5Config(; merge(merge((;), config), (is_decoder = false, is_encoder_decoder = false))...)
    encoder = HGFT5Stack(enc_config, shared)
    return HGFT5EncoderModel(shared, encoder)
end

# load model utils

basemodelkey(::HGFT5PreTrainedModel) = :transformer
basemodel(m::HGFT5PreTrainedModel) = m
isbasemodel(m::HGFT5PreTrainedModel) = true

get_model_type(::Val{:t5}) = (
    :model => HGFT5Model,
    :forconditionalgeneration => HGFT5ForConditionalGeneration,
    :encodermodel => HGFT5EncoderModel,
    :withlmheadmodel => HGFT5ForConditionalGeneration,
)

for (name, type) in get_model_type(:t5)
    @eval get_model_type(::Val{:t5}, ::Val{$(Meta.quot(name))}) = $type
end

is_seq2seq(::HGFT5Model) = true
is_seq2seq(::HGFT5ForConditionalGeneration) = true
