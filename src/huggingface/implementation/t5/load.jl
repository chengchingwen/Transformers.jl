using ..Layers
using ..Layers: CompositeEmbedding
using Functors

using ChainRulesCore
using NeuralAttentionlib
using NeuralAttentionlib: $, AbstractAttenOp, WithScore, t5_causal_bucketed_position_id, t5_bucketed_position_id,
    dot_product_score, masked_score, normalized_score, biased_score, dropout_score, weighted_sum_mixing,
    generic_multihead_qkv_attention


function _t5_attention(mask, p, bias = nothing)
    scoref = normalized_score(softmax) $ masked_score(NeuralAttentionlib.GenericMaskOp(), mask)
    scoref = isnothing(p) ? scoref : dropout_score(p) $ scoref
    scoref = isnothing(bias) ? scoref : (scoref $ biased_score(bias))
    return scoref
end

function t5rpe_attention_score(
    n_bucket::Integer, max_distance::Int, causal::Bool,
    q::AbstractArray, k::AbstractArray, position_embedding::AbstractArray, mask, p
)
    score = dot_product_score(q, k)
    rpe_id = causal ?
        t5_causal_bucketed_position_id(n_bucket, max_distance) :
        t5_bucketed_position_id(n_bucket, max_distance)
    position_bias = NeuralAttentionlib.get_scalar_relative_position_embeddings(rpe_id, position_embedding, score)
    scoref = _t5_attention(mask, p, position_bias)
    attention_score = scoref(identity, score)
    return (; attention_score, position_bias)
end

function ChainRulesCore.rrule(
    config::RuleConfig, ::typeof(t5rpe_attention_score),
    n_bucket::Integer, max_distance::Int, causal::Bool,
    q::AbstractArray, k::AbstractArray, position_embedding::AbstractArray, mask, p
)
    score, score_pullback = rrule(config, dot_product_score, q, k)
    rpe_id = causal ?
        t5_causal_bucketed_position_id(n_bucket, max_distance) :
        t5_bucketed_position_id(n_bucket, max_distance)
    position_bias, rpe_pullback = rrule(
        config, NeuralAttentionlib.get_scalar_relative_position_embeddings, rpe_id, position_embedding, score)
    scoref = _t5_attention(mask, p, position_bias)
    attention_score, attention_pullback = rrule(config, scoref, identity, score)
    function t5rpe_attention_score_pullback(Ȳ)
        ∂pf, _, ∂score = attention_pullback(Ȳ.attention_score)
        if ∂pf isa NoTangent
            ∂bias = Ȳ.position_bias
        else
            ∂bias = unthunk(Ȳ.position_bias) + unthunk(∂pf.arg[end])
        end
        ∂emb = rpe_pullback(∂bias)[3]
        _, ∂q, ∂k = score_pullback(∂score)
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), ∂q, ∂k, ∂emb, NoTangent(), NoTangent())
    end
    return (; attention_score, position_bias), t5rpe_attention_score_pullback
end

t5mixing(nt, v) = NamedTuple{(:hidden_state, :position_bias)}(t5mixing_ws(nt, v))
function t5mixing_ws(nt, v)
    s = nt.attention_score
    y = weighted_sum_mixing(s, v)
    return (hidden_state = y, attention_score = NeuralAttentionlib.unwrap_collapse(s), position_bias = nt.position_bias)
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(t5mixing), nt, v)
    nt2, mixing_pullback = rrule(config, t5mixing_ws, nt, v)
    return NamedTuple{(:hidden_state, :position_bias)}(nt2), mixing_pullback
end
function ChainRulesCore.rrule(config::RuleConfig, ::typeof(t5mixing_ws), nt, v)
    s = nt.attention_score
    bias = nt.position_bias
    y, mixing_pullback = rrule(config, weighted_sum_mixing, s, v)
    s′, unwrap_pullback = rrule(config, NeuralAttentionlib.unwrap_collapse, s)
    function t5mixing_pullback(Ȳ)
        _, ∂s1 = unwrap_pullback(Ȳ.attention_score)
        _, ∂s2, ∂v = mixing_pullback(Ȳ.hidden_state)
        ∂s = unthunk(∂s1) + unthunk(∂s2)
        ∂nt = (attention_score = ∂s, position_bias = Ȳ.position_bias)
        return (NoTangent(), Tangent{Any, typeof(∂nt)}(∂nt), ∂v)
    end
    return (; hidden_state = y, attention_score = s′, position_bias = bias), t5mixing_pullback
end

function t5rpe_multihead_qkv_attention(
    head::Integer, n_bucket::Integer, max_distance::Int, causal::Bool,
    q::AbstractArray, k::AbstractArray, v::AbstractArray, position_embedding::AbstractArray,
    mask = nothing, p = nothing
)
    return generic_multihead_qkv_attention(
        t5mixing, t5rpe_attention_score $ n_bucket $ max_distance $ causal,
        head, q, k, v, position_embedding, mask, p
    )
end

function t5rpe_multihead_qkv_attention(
    ::typeof(NeuralAttentionlib.score_returning),
    head::Integer, n_bucket::Integer, max_distance::Int, causal::Bool,
    q::AbstractArray, k::AbstractArray, v::AbstractArray, position_embedding::AbstractArray,
    mask = nothing, p = nothing
)
    return generic_multihead_qkv_attention(
        t5mixing_ws, t5rpe_attention_score $ n_bucket $ max_distance $ causal,
        head, q, k, v, position_embedding, mask, p
    )
end

function t5_multihead_qkv_attention(
    head::Integer, q::AbstractArray, k::AbstractArray, v::AbstractArray, bias::Union{Nothing, AbstractArray},
    mask = nothing, p = nothing
)
    return generic_multihead_qkv_attention(
        weighted_sum_mixing,
        _t5_attention(mask, p, bias) $
        dot_product_score,
        head, q, k, v)
end

function t5_multihead_qkv_attention(
    ::typeof(NeuralAttentionlib.score_returning),
    head::Integer, q::AbstractArray, k::AbstractArray, v::AbstractArray, bias::Union{Nothing, AbstractArray},
    mask = nothing, p = nothing
)
    return generic_multihead_qkv_attention(
        NeuralAttentionlib.score_returning(weighted_sum_mixing),
        _t5_attention(mask, p, bias) $
        dot_product_score,
        head, q, k, v)
end

struct T5RPEMultiheadQKVAttenOp{F, E} <: AbstractAttenOp
    head::Int
    n_bucket::Int
    max_distance::Int
    position_embedding::E
    p::F
end
@functor T5RPEMultiheadQKVAttenOp (position_embedding,)

T5RPEMultiheadQKVAttenOp(head, n_bucket, max_distance, position_embedding) =
    T5RPEMultiheadQKVAttenOp(head, n_bucket, max_distance, position_embedding, nothing)

NeuralAttentionlib.get_attention_func(::T5RPEMultiheadQKVAttenOp) = t5rpe_multihead_qkv_attention
NeuralAttentionlib.get_attention_func_args(op::T5RPEMultiheadQKVAttenOp, q, k, v, mask = nothing) =
    (op.head, op.n_bucket, op.max_distance, false, q, k, v, op.position_embedding,
     NeuralAttentionlib.BatchedMask(mask), op.p)

function Base.show(io::IO, op::T5RPEMultiheadQKVAttenOp)
    print(io, "T5RPEMultiheadQKVAttenOp(head = ", op.head, ", n_bucket = ", op.n_bucket,
          ", max_distance = ", op.max_distance, ", position_embedding = ", reverse(size(op.position_embedding)),
          ", p = ", op.p, ')')
end

const T5RPEMultiheadQKVAttenOpWithScore{F, E} = WithScore{T5RPEMultiheadQKVAttenOp{F, E}}
function Functors.functor(::Type{<:T5RPEMultiheadQKVAttenOpWithScore}, op)
    head, n_bucket, max_distance, p = op.head, op.n_bucket, op.max_distance, op.p
    return ((position_embedding = op.position_embedding,),
            y -> T5RPEMultiheadQKVAttenOpWithScore(head, n_bucket, max_distance, y.position_embedding, p))
end

struct T5RPECausalMultiheadQKVAttenOp{F, E} <: AbstractAttenOp
    head::Int
    n_bucket::Int
    max_distance::Int
    position_embedding::E
    p::F
end
@functor T5RPECausalMultiheadQKVAttenOp (position_embedding,)

T5RPECausalMultiheadQKVAttenOp(head, n_bucket, max_distance, position_embedding) =
    T5RPECausalMultiheadQKVAttenOp(head, n_bucket, max_distance, position_embedding, nothing)

NeuralAttentionlib.get_attention_func(::T5RPECausalMultiheadQKVAttenOp) = t5rpe_multihead_qkv_attention
NeuralAttentionlib.get_attention_func_args(op::T5RPECausalMultiheadQKVAttenOp, q, k, v, mask = nothing) =
    (op.head, op.n_bucket, op.max_distance, true, q, k, v, op.position_embedding,
     NeuralAttentionlib.BatchedMask(mask), op.p)

function Base.show(io::IO, op::T5RPECausalMultiheadQKVAttenOp)
    print(io, "T5RPECausalMultiheadQKVAttenOp(head = ", op.head, ", n_bucket = ", op.n_bucket,
          ", max_distance = ", op.max_distance, ", position_embedding = ", reverse(size(op.position_embedding)),
          ", p = ", op.p, ')')
end

const T5RPECausalMultiheadQKVAttenOpWithScore{F, E} = WithScore{T5RPECausalMultiheadQKVAttenOp{F, E}}
function Functors.functor(::Type{<:T5RPECausalMultiheadQKVAttenOpWithScore}, op)
    head, n_bucket, max_distance, p = op.head, op.n_bucket, op.max_distance, op.p
    return ((position_embedding = op.position_embedding,),
            y -> T5RPECausalMultiheadQKVAttenOpWithScore(head, n_bucket, max_distance, y.position_embedding, p))
end

struct T5BiasedMultiheadQKVAttenOp{F} <: AbstractAttenOp
    head::Int
    p::F
end
T5BiasedMultiheadQKVAttenOp(head) = T5BiasedMultiheadQKVAttenOp(head, nothing)
NeuralAttentionlib.get_attention_func(::T5BiasedMultiheadQKVAttenOp) = t5_multihead_qkv_attention
NeuralAttentionlib.get_attention_func_args(op::T5BiasedMultiheadQKVAttenOp, q, k, v, bias, mask = nothing) =
    (op.head, q, k, v, bias, NeuralAttentionlib.BatchedMask(mask), op.p)

const T5BiasedMultiheadQKVAttenOpWithScore{F} = WithScore{T5BiasedMultiheadQKVAttenOp{F}}

struct T5MultiheadQKVAttenOp{F} <: AbstractAttenOp
    head::Int
    p::F
end
T5MultiheadQKVAttenOp(head) = T5MultiheadQKVAttenOp(head, nothing)
NeuralAttentionlib.get_attention_func(::T5MultiheadQKVAttenOp) = t5_multihead_qkv_attention
NeuralAttentionlib.get_attention_func_args(op::T5MultiheadQKVAttenOp, q, k, v, mask = nothing) =
    (op.head, q, k, v, nothing, NeuralAttentionlib.BatchedMask(mask), op.p)

const T5MultiheadQKVAttenOpWithScore{F} = WithScore{T5MultiheadQKVAttenOp{F}}

Layers.argument_names(
    ::Union{T5RPEMultiheadQKVAttenOpWithScore, T5RPEMultiheadQKVAttenOp,
            T5RPECausalMultiheadQKVAttenOpWithScore, T5RPECausalMultiheadQKVAttenOp,
            T5MultiheadQKVAttenOpWithScore, T5MultiheadQKVAttenOp}
) = (:hidden_state, :attention_mask)
Layers.argument_names(::T5BiasedMultiheadQKVAttenOpWithScore) = (:hidden_state, :attention_mask, :position_bias)
Layers.argument_names(::T5BiasedMultiheadQKVAttenOp) = (:hidden_state, :attention_mask, :position_bias)

function Layers.apply_on_namedtuple(
    op::Union{T5RPEMultiheadQKVAttenOpWithScore, T5RPEMultiheadQKVAttenOp,
              T5RPECausalMultiheadQKVAttenOpWithScore, T5RPECausalMultiheadQKVAttenOp,
              T5MultiheadQKVAttenOpWithScore, T5MultiheadQKVAttenOp}, nt::NamedTuple
)
    return Layers.apply_attention_op(op, nt)
end

function Layers.apply_on_namedtuple(
    op::Union{T5BiasedMultiheadQKVAttenOpWithScore, T5BiasedMultiheadQKVAttenOp}, nt::NamedTuple
)
    qkv = nt.hidden_state
    qkv isa NTuple{3, Any} ||
        error("Expect hidden_state to be a tuple of 3 arrays, but get $(typeof(qkv)).")
    mask = get(nt, :attention_mask, nothing)
    bias = nt.position_bias
    a = op(qkv..., bias, mask)
    return Layers.return_hidden_state(nt, a)
end

struct T5Gated{G, L}
    gate::G
    linear::L
end
@functor T5Gated

(g::T5Gated)(x) = g.gate(x) .* g.linear(x)

abstract type HGFT5PreTrainedModel <: HGFPreTrainedModel end

struct HGFT5Model{E, S} <: HGFT5PreTrainedModel
    embed::E
    seq2seq::S
end
@functor HGFT5Model

function (model::HGFT5Model)(nt::NamedTuple)
    embs = model.embed(nt)
    outputs = model.seq2seq(embs)
    encoder_output = Base.structdiff(outputs.encoder_output, NamedTuple{(:position_bias,)})
    decoder_output = Base.structdiff(outputs.decoder_output, NamedTuple{(:position_bias,)})
    return merge(outputs, (; encoder_output, decoder_output))
end

struct HGFT5ForConditionalGeneration{M, H} <: HGFT5PreTrainedModel
    model::M
    lm_head::H
end
@functor HGFT5ForConditionalGeneration

function (model::HGFT5ForConditionalGeneration)(nt::NamedTuple)
    outputs = model.model(nt)
    outputs = model.lm_head(outputs)
    return outputs
end

struct HGFT5EncoderModel{E, ENC} <: HGFT5PreTrainedModel
    embed::E
    encoder::ENC
end
@functor HGFT5EncoderModel

function (model::HGFT5EncoderModel)(nt::NamedTuple)
    outputs = model.encoder(model.embed(nt))
    return Base.structdiff(outputs, NamedTuple{(:position_bias,)})
end

for T in :[
    HGFT5Model, HGFT5ForConditionalGeneration, HGFT5EncoderModel
].args
    @eval function Base.show(io::IO, m::MIME"text/plain", x::$T)
        if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
            Flux._big_show(io, x)
        elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
            Flux._layer_show(io, x)
        else
            show(io, x)
        end
    end
end

get_model_type(::Val{:t5}) = (
    :model => HGFT5Model,
    :forconditionalgeneration => HGFT5ForConditionalGeneration,
    :encodermodel => HGFT5EncoderModel,
    :withlmheadmodel => HGFT5ForConditionalGeneration,
)

for (name, type) in get_model_type(Val(:t5))
    @eval get_model_type(::Val{:t5}, ::Val{$(Meta.quot(name))}) = $type
end

is_seq2seq(::HGFT5Model) = true
is_seq2seq(::HGFT5ForConditionalGeneration) = true


function t5_weight_init(din, dout, factor = true)
    function weight_init()
        weight = randn(Float32, dout, din)
        if !isone(factor)
            weight .*= factor
        end
        return weight
    end
    return weight_init
end

function load_model(_type::Type{<:HGFT5Model}, cfg, state_dict = OrderedDict{String, Any}(), prefix = "")
    embed = load_model(_type, CompositeEmbedding, cfg, state_dict, joinname(prefix, "shared"))
    seq2seq = load_model(_type, Seq2Seq, cfg, state_dict, prefix)
    return HGFT5Model(Layers.Parallel{(:encoder_input, :decoder_input)}(embed), seq2seq)
end

function load_model(::Type{<:HGFT5ForConditionalGeneration}, cfg,
                    state_dict = OrderedDict{String, Any}(), prefix = "")
    model = load_model(HGFT5Model, cfg, state_dict, prefix)
    if cfg[:tie_word_embeddings]
        embedding = model.embed.layer.token.embeddings
        scale = convert(eltype(embedding), inv(sqrt(size(embedding, 1))))
    else
        vocab_size, dims, factor = cfg[:vocab_size], cfg[:d_model], Float32(cfg[:initializer_factor])
        embedding = getweight(t5_weight_init(vocab_size, dims, factor), Layers.Embed,
                              state_dict, joinname(prefix, "lm_head.weight"))
        scale = nothing
    end
    lm_head = Layers.EmbedDecoder(Layers.Embed(scale, embedding))
    return HGFT5ForConditionalGeneration(model, Layers.Branch{:logit, (:hidden_state,)}(lm_head))
end

function load_model(_type::Type{<:HGFT5EncoderModel}, cfg,
                    state_dict = OrderedDict{String, Any}(), prefix = "")
    embed = load_model(HGFT5Model, CompositeEmbedding, cfg, state_dict, prefix)
    encoder = load_model(HGFT5Model, TransformerBlock, cfg, state_dict, joinname(prefix, "encoder"))
    return HGFT5EncoderModel(embed, encoder)
end

function load_model(::Type{<:HGFT5Model}, ::Type{<:CompositeEmbedding}, cfg, state_dict, prefix)
    vocab_size, dims, factor = cfg[:vocab_size], cfg[:d_model], Float32(cfg[:initializer_factor])
    weight = getweight(t5_weight_init(vocab_size, dims, factor), Layers.Embed, state_dict, joinname(prefix, "weight"))
    embed = CompositeEmbedding(token = Embed(nothing, weight))
    return embed
end

function load_model(_type::Type{<:HGFT5Model}, ::Type{<:Seq2Seq}, cfg, state_dict, prefix)
    encoder = load_model(_type, TransformerBlock, cfg, state_dict, joinname(prefix, "encoder"))
    decoder = load_model(_type, TransformerDecoderBlock, cfg, state_dict, joinname(prefix, "decoder"))
    return Seq2Seq(encoder, decoder)
end

function load_model(::Type{<:HGFT5Model}, ::Type{<:Layers.RMSLayerNorm}, cfg, state_dict, prefix)
    dims = cfg[:d_model]
    ln_ϵ = Float32(cfg[:layer_norm_epsilon])
    ln_init = () -> ones(Float32, dims)
    ln_weight = getweight(ln_init, Array, state_dict, joinname(prefix, "weight"))
    return Layers.RMSLayerNorm(ln_weight, ln_ϵ)
end

t5_collect_outputs(prev, output) = merge(output, Layers.collect_outputs(prev, Base.structdiff(output, NamedTuple{(:position_bias,)})))

function load_model(
    ::Type{<:HGFT5Model}, ::Type{<:Layers.SelfAttention{A}}, cfg, state_dict, prefix
) where {A <: Union{T5RPEMultiheadQKVAttenOp, T5RPECausalMultiheadQKVAttenOp, T5BiasedMultiheadQKVAttenOp}}
    dims, head, kv_dims = cfg[:d_model], cfg[:num_heads], cfg[:d_kv]
    rpe_nbucket, rpe_max_dist = cfg[:relative_attention_num_buckets], cfg[:relative_attention_max_distance]
    p = Float64(cfg[:dropout_rate]); p = iszero(p) ? nothing : p
    factor = Float32(cfg[:initializer_factor])
    return_score = cfg[:output_attentions]
    q_init = t5_weight_init(dims, head * kv_dims, factor / sqrt(dims * kv_dims))
    kv_init = t5_weight_init(dims, head * kv_dims, factor / sqrt(dims))
    o_init = t5_weight_init(dims, head * kv_dims, factor / sqrt(head * kv_dims))
    q_weight = getweight(q_init,  Array, state_dict, joinname(prefix, "q.weight"))
    k_weight = getweight(kv_init, Array, state_dict, joinname(prefix, "k.weight"))
    v_weight = getweight(kv_init, Array, state_dict, joinname(prefix, "v.weight"))
    o_weight = getweight(o_init,  Array, state_dict, joinname(prefix, "o.weight"))
    qkv_proj = Layers.Fork(Layers.Dense(q_weight), Layers.Dense(k_weight), Layers.Dense(v_weight))
    o_proj = Layers.Dense(o_weight)
    if A <: Union{T5RPEMultiheadQKVAttenOp, T5RPECausalMultiheadQKVAttenOp}
        rpe_weight = getweight(t5_weight_init(rpe_nbucket, head, factor / sqrt(dims)), Layers.Embed,
                               state_dict, joinname(prefix, "relative_attention_bias.weight"))
        if A <: T5RPEMultiheadQKVAttenOp
            op = T5RPEMultiheadQKVAttenOp(head, rpe_nbucket, rpe_max_dist, rpe_weight, p)
        else
            op = T5RPECausalMultiheadQKVAttenOp(head, rpe_nbucket, rpe_max_dist, rpe_weight, p)
        end
    else
        op = T5BiasedMultiheadQKVAttenOp(head, p)
    end
    return_score && (op = WithScore(op))
    return Layers.SelfAttention(op, qkv_proj, o_proj)
end

function load_model(::Type{<:HGFT5Model}, ::Type{<:Layers.CrossAttention}, cfg, state_dict, prefix)
    dims, head, kv_dims = cfg[:d_model], cfg[:num_heads], cfg[:d_kv]
    p = Float64(cfg[:dropout_rate]); p = iszero(p) ? nothing : p
    factor = Float32(cfg[:initializer_factor])
    return_score = cfg[:output_attentions]
    q_init = t5_weight_init(dims, head * kv_dims, factor / sqrt(dims * kv_dims))
    kv_init = t5_weight_init(dims, head * kv_dims, factor / sqrt(dims))
    o_init = t5_weight_init(dims, head * kv_dims, factor / sqrt(head * kv_dims))
    q_weight = getweight(q_init,  Array, state_dict, joinname(prefix, "q.weight"))
    k_weight = getweight(kv_init, Array, state_dict, joinname(prefix, "k.weight"))
    v_weight = getweight(kv_init, Array, state_dict, joinname(prefix, "v.weight"))
    o_weight = getweight(o_init,  Array, state_dict, joinname(prefix, "o.weight"))
    q_proj =  Layers.Dense(q_weight)
    kv_proj = Layers.Fork(Layers.Dense(k_weight), Layers.Dense(v_weight))
    o_proj = Layers.Dense(o_weight)
    op = T5MultiheadQKVAttenOp(head, p)
    return_score && (op = WithScore(op))
    return Layers.CrossAttention(op, q_proj, kv_proj, o_proj)
end

function load_model(
    ::Type{<:HGFT5Model}, ::Type{Layers.Chain{Tuple{Layers.Dense, Layers.Dense}}},
    cfg, state_dict, prefix
)
    dims, ff_dims = cfg[:d_model], cfg[:d_ff]
    factor = Float32(cfg[:initializer_factor])
    p = Float64(cfg[:dropout_rate]); p = iszero(p) ? nothing : p
    isgated = cfg[:is_gated_act]
    act = ACT2FN[Symbol(cfg[:dense_act_fn])]
    wi_init = t5_weight_init(dims, ff_dims, factor / sqrt(dims))
    wo_init = t5_weight_init(ff_dims, dims, factor / sqrt(ff_dims))
    if isgated
        wi0_weight = getweight(wi_init, Array, state_dict, joinname(prefix, "wi0.weight"))
        wi1_weight = getweight(wi_init, Array, state_dict, joinname(prefix, "wi1.weight"))
        wi = T5Gated(Layers.Dense(act, wi0_weight), Layers.Dense(wi1_weight))
    else
        wi_weight = getweight(wi_init, Array, state_dict, joinname(prefix, "wi.weight"))
        wi = Layers.Dense(act, wi_weight)
    end
    wo_weight = getweight(wo_init, Array, state_dict, joinname(prefix, "wo.weight"))
    return Layers.Chain(Layers.DropoutLayer(wi, p), Layers.Dense(wo_weight))
end

function load_model(_type::Type{<:HGFT5Model}, ::Type{<:TransformerBlock}, cfg, state_dict, prefix)
    n = cfg[:num_layers]
    p = Float64(cfg[:dropout_rate]); p = iszero(p) ? nothing : p
    collect_output = cfg[:output_attentions] || cfg[:output_hidden_states]
    blocks = []
    for i = 1:n
        lprefix = joinname(prefix, :block, i-1, :layer)
        sa_ln = load_model(_type, Layers.RMSLayerNorm, cfg, state_dict, joinname(lprefix, "0.layer_norm"))
        op_type = isone(i) ? T5RPEMultiheadQKVAttenOp : T5BiasedMultiheadQKVAttenOp
        sa = load_model(_type, Layers.SelfAttention{op_type}, cfg, state_dict, joinname(lprefix, "0.SelfAttention"))
        sa = Layers.PreNormResidual(Layers.DropoutLayer(sa, p), sa_ln)
        ff_ln = load_model(_type, Layers.RMSLayerNorm, cfg, state_dict, joinname(lprefix, "1.layer_norm"))
        ff = load_model(_type, Layers.Chain{Tuple{Layers.Dense, Layers.Dense}},
                        cfg, state_dict, joinname(lprefix, "1.DenseReluDense"))
        ff = Layers.PreNormResidual(Layers.DropoutLayer(ff, p), ff_ln)
        block = TransformerBlock(sa, ff)
        push!(blocks, block)
    end
    collect_f = collect_output ? t5_collect_outputs : nothing
    trf = Transformer(Tuple(blocks), collect_f)
    final_ln = load_model(_type, Layers.RMSLayerNorm, cfg, state_dict, joinname(prefix, "final_layer_norm"))
    return Layers.Chain(trf, final_ln)
end

function load_model(_type::Type{<:HGFT5Model}, ::Type{<:TransformerDecoderBlock}, cfg, state_dict, prefix)
    n = cfg[:num_layers]
    p = Float64(cfg[:dropout_rate]); p = iszero(p) ? nothing : p
    collect_output = cfg[:output_attentions] || cfg[:output_hidden_states]
    blocks = []
    for i = 1:n
        lprefix = joinname(prefix, :block, i-1, :layer)
        sa_ln = load_model(_type, Layers.RMSLayerNorm, cfg, state_dict, joinname(lprefix, "0.layer_norm"))
        op_type = isone(i) ? T5RPECausalMultiheadQKVAttenOp : T5BiasedMultiheadQKVAttenOp
        sa = load_model(_type, Layers.SelfAttention{op_type}, cfg, state_dict, joinname(lprefix, "0.SelfAttention"))
        sa = Layers.PreNormResidual(Layers.DropoutLayer(sa, p), sa_ln)
        ca_ln = load_model(_type, Layers.RMSLayerNorm, cfg, state_dict, joinname(lprefix, "1.layer_norm"))
        ca = load_model(_type, Layers.CrossAttention, cfg, state_dict, joinname(lprefix, "1.EncDecAttention"))
        ca = Layers.PreNormResidual(Layers.DropoutLayer(ca, p), ca_ln)
        ff_ln = load_model(_type, Layers.RMSLayerNorm, cfg, state_dict, joinname(lprefix, "2.layer_norm"))
        ff = load_model(_type, Layers.Chain{Tuple{Layers.Dense, Layers.Dense}},
                        cfg, state_dict, joinname(lprefix, "2.DenseReluDense"))
        ff = Layers.PreNormResidual(Layers.DropoutLayer(ff, p), ff_ln)
        block = TransformerDecoderBlock(sa, ca, ff)
        push!(blocks, block)
    end
    collect_f = collect_output ? t5_collect_outputs : nothing
    trf = Transformer(Tuple(blocks), collect_f)
    final_ln = load_model(_type, Layers.RMSLayerNorm, cfg, state_dict, joinname(prefix, "final_layer_norm"))
    return Layers.Chain(trf, final_ln)
end


function get_state_dict(m::HGFT5Model, state_dict = OrderedDict{String, Any}(), prefix = "")
    get_state_dict(HGFT5Model, m.embed.layer, state_dict, joinname(prefix, "shared"))
    get_state_dict(HGFT5Model, m.seq2seq, state_dict, prefix)
    return state_dict
end

function get_state_dict(m::HGFT5ForConditionalGeneration, state_dict = OrderedDict{String, Any}(), prefix = "")
    get_state_dict(m.model, state_dict, prefix)
    embedding = m.lm_head.layer.embed.embeddings
    state_dict[joinname(prefix, "lm_head.weight")] = embedding'
    return state_dict
end

function get_state_dict(m::HGFT5EncoderModel, state_dict = OrderedDict{String, Any}(), prefix = "")
    get_state_dict(HGFT5Model, m.embed, state_dict, joinname(prefix, "shared"))
    get_state_dict(HGFT5Model, m.encoder[1], state_dict, joinname(prefix, "encoder"))
    get_state_dict(HGFT5Model, m.encoder[2], state_dict, joinname(prefix, "encoder.final_layer_norm"))
    return state_dict
end

get_state_dict(p::Type{<:HGFT5Model}, m::CompositeEmbedding, state_dict, prefix) = get_state_dict(p, m.token, state_dict, prefix)

function get_state_dict(p::Type{<:HGFT5Model}, m::Seq2Seq, state_dict, prefix)
    get_state_dict(p, m.encoder[1], state_dict, joinname(prefix, "encoder"))
    get_state_dict(p, m.encoder[2], state_dict, joinname(prefix, "encoder.final_layer_norm"))
    get_state_dict(p, m.decoder[1], state_dict, joinname(prefix, "decoder"))
    get_state_dict(p, m.decoder[2], state_dict, joinname(prefix, "decoder.final_layer_norm"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFT5Model}, m::Layers.RMSLayerNorm, state_dict, prefix)
    state_dict[joinname(prefix, "weight")] = m.α
    return state_dict
end

function get_state_dict(p::Type{<:HGFT5Model}, m::Layers.SelfAttention, state_dict, prefix)
    get_state_dict(p, m.qkv_proj.layers[1], state_dict, joinname(prefix, "q"))
    get_state_dict(p, m.qkv_proj.layers[2], state_dict, joinname(prefix, "k"))
    get_state_dict(p, m.qkv_proj.layers[3], state_dict, joinname(prefix, "v"))
    get_state_dict(p, m.o_proj, state_dict, joinname(prefix, "o"))
    if m.attention_op isa Union{
        T5RPEMultiheadQKVAttenOp, T5RPECausalMultiheadQKVAttenOp,
        T5RPEMultiheadQKVAttenOpWithScore, T5RPECausalMultiheadQKVAttenOpWithScore
    }
        state_dict[joinname(prefix, "relative_attention_bias.weight")] = m.attention_op.position_embedding'
    end
    return state_dict
end

function get_state_dict(p::Type{<:HGFT5Model}, m::Layers.CrossAttention, state_dict, prefix)
    get_state_dict(p, m.q_proj, state_dict, joinname(prefix, "q"))
    get_state_dict(p, m.kv_proj.layers[1], state_dict, joinname(prefix, "k"))
    get_state_dict(p, m.kv_proj.layers[2], state_dict, joinname(prefix, "v"))
    get_state_dict(p, m.o_proj, state_dict, joinname(prefix, "o"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFT5Model}, m::Layers.Chain{<:Tuple{Any, Layers.Dense}}, state_dict, prefix)
    if m[1] isa T5Gated
        get_state_dict(p, m[1].layer.gate, state_dict, joinname(prefix, "wi0"))
        get_state_dict(p, m[1].layer.linear, state_dict, joinname(prefix, "wi1"))
    else
        get_state_dict(p, m[1], state_dict, joinname(prefix, "wi"))
    end
    get_state_dict(p, m[2], state_dict, joinname(prefix, "wo"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFT5Model}, m::Transformer, state_dict, prefix)
    for (i, t) in enumerate(m.blocks)
        get_state_dict(p, t, state_dict, joinname(prefix, :block, i-1, :layer))
    end
    return state_dict
end

function get_state_dict(p::Type{<:HGFT5Model}, m::TransformerBlock, state_dict, prefix)
    get_state_dict(p, m.attention.layer, state_dict, joinname(prefix, "0.SelfAttention"))
    get_state_dict(p, m.attention.norm, state_dict, joinname(prefix, "0.layer_norm"))
    get_state_dict(p, m.feedforward.layer, state_dict, joinname(prefix, "1.DenseReluDense"))
    get_state_dict(p, m.feedforward.norm, state_dict, joinname(prefix, "1.layer_norm"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFT5Model}, m::TransformerDecoderBlock, state_dict, prefix)
    get_state_dict(p, m.attention.layer, state_dict, joinname(prefix, "0.SelfAttention"))
    get_state_dict(p, m.attention.norm, state_dict, joinname(prefix, "0.layer_norm"))
    get_state_dict(p, m.crossattention.layer, state_dict, joinname(prefix, "1.EncDecAttention"))
    get_state_dict(p, m.crossattention.norm, state_dict, joinname(prefix, "1.layer_norm"))
    get_state_dict(p, m.feedforward.layer, state_dict, joinname(prefix, "2.DenseReluDense"))
    get_state_dict(p, m.feedforward.norm, state_dict, joinname(prefix, "2.layer_norm"))
    return state_dict
end
