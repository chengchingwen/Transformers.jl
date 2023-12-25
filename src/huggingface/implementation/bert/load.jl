using ..Layers
using ..Layers: CompositeEmbedding, SelfAttention, MultiheadQKVAttenOp
using ChainRulesCore
using Functors
using FillArrays
using NNlib
using Static

using NeuralAttentionlib
using NeuralAttentionlib: WithScore

struct BertPooler{D}
    dense::D
end
@functor BertPooler
@fluxlayershow BertPooler

function (m::BertPooler)(x)
    first_token = selectdim(x, 2, 1)
    return m.dense(first_token)
end

struct BertQA{D}
    dense::D
end
@functor BertQA
@fluxlayershow BertQA

function _slice(x, i)
    cs = ntuple(i->Colon(), static(ndims(x)) - static(1))
    @view x[i, cs...]
end

function (m::BertQA)(x)
    logits = m.dense(x)
    start_logit = _slice(logits, 1)
    end_logit = _slice(logits, 2)
    return (; start_logit, end_logit)
end
(m::BertQA)(nt::NamedTuple) = merge(nt, m(nt.hidden_state))

@hgfdef Bert (
    Model => begin
      outputs = model.encoder(model.embed(nt))
      if isnothing(model.pooler)
        return outputs
      else
        return model.pooler(outputs)
      end
    end,
    ForPreTraining,
    LMHeadModel,
    ForMaskedLM,
    ForNextSentencePrediction,
    ForSequenceClassification,
    ForTokenClassification,
    ForQuestionAnswering,
    # ForMultipleChoice,
)

basemodelkey(::Type{<:HGFPreTrained{:bert}}) = :bert

bert_ones_like(x::AbstractArray) = Ones{Int}(Base.tail(size(x)))
ChainRulesCore.@non_differentiable bert_ones_like(x)

load_model(_type::Type{HGFBertModel}, cfg, state_dict, prefix) =
    load_model(_type, _type, cfg, state_dict, prefix)
function load_model(_type::Type, ::Type{HGFBertModel}, cfg, state_dict, prefix)
    embed = load_model(_type, CompositeEmbedding, cfg, state_dict, joinname(prefix, "embeddings"))
    encoder = load_model(_type, TransformerBlock,  cfg, state_dict, joinname(prefix, "encoder"))
    dims = cfg[:hidden_size]
    factor = Float32(cfg[:initializer_range])
    weight = getweight(weight_init(dims, dims, factor), Array, state_dict, joinname(prefix, "pooler.dense.weight"))
    bias = getweight(zero_init(dims), Array, state_dict, joinname(prefix, "pooler.dense.bias"))
    pooler = BertPooler(Layers.Dense(NNlib.tanh_fast, weight, bias))
    return HGFBertModel(embed, encoder, Layers.Branch{(:pooled,), (:hidden_state,)}(pooler))
end
function load_model(
    _type::Type{<:Union{
        HGFBertLMHeadModel, HGFBertForMaskedLM,
        HGFBertForTokenClassification, HGFBertForQuestionAnswering,
    }},
    ::Type{HGFBertModel}, cfg, state_dict, prefix)
    embed = load_model(_type, CompositeEmbedding, cfg, state_dict, joinname(prefix, "embeddings"))
    encoder = load_model(_type, TransformerBlock,  cfg, state_dict, joinname(prefix, "encoder"))
    return HGFBertModel(embed, encoder, nothing)
end

function load_model(_type::Type{HGFBertForPreTraining}, cfg, state_dict, prefix)
    bert = load_model(_type, HGFBertLMHeadModel, cfg, state_dict, prefix)
    model, lmhead = bert.model, bert.cls
    seqhead = load_model(HGFBertForNextSentencePrediction, Layers.Dense, cfg, state_dict, prefix)
    cls = Layers.Chain(lmhead, seqhead)
    return HGFBertForPreTraining(model, cls)
end

load_model(_type::Type{HGFBertLMHeadModel}, cfg, state_dict, prefix) =
    load_model(_type, _type, cfg, state_dict, prefix)
function load_model(_type::Type, ::Type{HGFBertLMHeadModel}, cfg, state_dict, prefix)
    model = load_model(_type, HGFBertModel, cfg, state_dict, joinname(prefix, "bert"))
    dims, vocab_size, pad_id = cfg[:hidden_size], cfg[:vocab_size], cfg[:pad_token_id]
    factor = Float32(cfg[:initializer_range])
    act = ACT2FN[Symbol(cfg[:hidden_act])]
    # HGFBertPredictionHeadTransform
    head_weight = getweight(weight_init(dims, dims, factor), Array,
                            state_dict, joinname(prefix, "cls.predictions.transform.dense.weight"))
    head_bias = getweight(zero_init(dims), Array,
                          state_dict, joinname(prefix, "cls.predictions.transform.dense.bias"))
    head_ln = load_model(HGFBertModel, Layers.LayerNorm, cfg,
                         state_dict, joinname(prefix, "cls.predictions.transform.LayerNorm"))
    # HGFBertLMPredictionHead
    if cfg[:tie_word_embeddings]
        embedding = model.embed[1].token.embeddings
    else
        embedding = getweight(Layers.Embed, state_dict, joinname(prefix, "cls.predictions.decoder.weight")) do
            weight = weight_init(vocab_size, dims, factor)()
            weight[:, pad_id+1] .= 0
            return weight
        end
    end
    bias = getweight(zero_init(vocab_size), Array, state_dict, joinname(prefix, "cls.predictions.bias"))
    lmhead = Layers.Chain(Layers.Dense(act, head_weight, head_bias), head_ln,
                          Layers.EmbedDecoder(Layers.Embed(embedding), bias))
    return HGFBertLMHeadModel(model, Layers.Branch{(:logit,), (:hidden_state,)}(lmhead))
end

function load_model(_type::Type{HGFBertForMaskedLM}, cfg, state_dict, prefix)
    bert = load_model(HGFBertLMHeadModel, cfg, state_dict, prefix)
    model, lmhead = bert.model, bert.cls
    return HGFBertForMaskedLM(model, lmhead)
end

function load_model(_type::Type{HGFBertForNextSentencePrediction}, cfg, state_dict, prefix)
    model = load_model(HGFBertModel, cfg, state_dict, joinname(prefix, "bert"))
    cls = load_model(HGFBertForNextSentencePrediction, Layers.Dense, cfg, state_dict, prefix)
    return HGFBertForNextSentencePrediction(model, cls)
end

function load_model(_type::Type{HGFBertForSequenceClassification}, cfg, state_dict, prefix)
    model = load_model(HGFBertModel, cfg, state_dict, joinname(prefix, "bert"))
    dims, nlabel = cfg[:hidden_size], cfg[:num_labels]
    factor = Float32(cfg[:initializer_range])
    weight = getweight(weight_init(dims, nlabel, factor), Array,
                       state_dict, joinname(prefix, "classifier.weight"))
    bias = getweight(zero_init(nlabel), Array, state_dict, joinname(prefix, "classifier.bias"))
    cls = Layers.Branch{(:logit,), (:pooled,)}(Layers.Dense(weight, bias))
    return HGFBertForSequenceClassification(model, cls)
end

# function load_model(
#     _type::Type{HGFBertForMultipleChoice}, cfg,
#     state_dict = OrderedDict{String, Any}(), prefix = ""
# )
#     model = load_model(HGFBertModel, cfg, state_dict, joinname(prefix, "bert"))

# end

function load_model(_type::Type{HGFBertForTokenClassification}, cfg, state_dict, prefix)
    model = load_model(_type, HGFBertModel, cfg, state_dict, joinname(prefix, "bert"))
    dims, nlabel = cfg[:hidden_size], cfg[:num_labels]
    factor = Float32(cfg[:initializer_range])
    weight = getweight(weight_init(dims, nlabel, factor), Array,
                       state_dict, joinname(prefix, "classifier.weight"))
    bias = getweight(zero_init(nlabel), Array, state_dict, joinname(prefix, "classifier.bias"))
    cls = Layers.Branch{(:logit,), (:hidden_state,)}(Layers.Dense(weight, bias))
    return HGFBertForTokenClassification(model, cls)
end

function load_model(_type::Type{HGFBertForQuestionAnswering}, cfg, state_dict, prefix)
    model = load_model(_type, HGFBertModel, cfg, state_dict, joinname(prefix, "bert"))
    dims, nlabel = cfg[:hidden_size], cfg[:num_labels]
    factor = Float32(cfg[:initializer_range])
    weight = getweight(weight_init(dims, nlabel, factor), Array,
                       state_dict, joinname(prefix, "qa_outputs.weight"))
    bias = getweight(zero_init(nlabel), Array, state_dict, joinname(prefix, "qa_outputs.bias"))
    cls = BertQA(Layers.Dense(weight, bias))
    return HGFBertForQuestionAnswering(model, cls)
end

function load_model(_type::Type{HGFBertForNextSentencePrediction}, ::Type{Layers.Dense}, cfg, state_dict, prefix)
    dims = cfg[:hidden_size]
    factor = Float32(cfg[:initializer_range])
    seq_weight = getweight(weight_init(dims, 2, factor), Array,
                           state_dict, joinname(prefix, "cls.seq_relationship.weight"))
    seq_bias = getweight(zero_init(2), Array, state_dict, joinname(prefix, "cls.seq_relationship.bias"))
    seqhead = Layers.Dense(seq_weight, seq_bias)
    return Layers.Branch{(:seq_logit,), (:pooled,)}(Layers.RenameArgs{(:hidden_state,), (:pooled,)}(seqhead))
end

function load_model(_type::Type{<:HGFBertPreTrainedModel}, ::Type{<:Layers.LayerNorm}, cfg, state_dict, prefix)
    dims = cfg[:hidden_size]
    ln_ϵ = Float32(cfg[:layer_norm_eps])
    old_weight_name = joinname(prefix, "gamma")
    old_bias_name = joinname(prefix, "beta")
    weight_name = haskey(state_dict, old_weight_name) ? old_weight_name : joinname(prefix, "weight")
    bias_name = haskey(state_dict, old_bias_name) ? old_bias_name : joinname(prefix, "bias")
    ln_weight = getweight(one_init(dims), Array, state_dict, weight_name)
    ln_bias = getweight(zero_init(dims), Array, state_dict, bias_name)
    return Layers.LayerNorm(ln_weight, ln_bias, ln_ϵ)
end

function load_model(_type::Type{<:HGFBertPreTrainedModel}, ::Type{<:CompositeEmbedding}, cfg, state_dict, prefix)
    vocab_size, dims, pad_id = cfg[:vocab_size], cfg[:hidden_size], cfg[:pad_token_id]
    max_pos, n_type = cfg[:max_position_embeddings], cfg[:type_vocab_size]
    p = cfg[:hidden_dropout_prob]; p = iszero(p) ? nothing : p
    pe_type = cfg[:position_embedding_type]
    pe_type == "absolute" || load_error("Right now only absolute PE is supported in Bert.")
    factor = Float32(cfg[:initializer_range])
    token_weight = getweight(Layers.Embed, state_dict, joinname(prefix, "word_embeddings.weight")) do
        weight = weight_init(vocab_size, dims, factor)()
        weight[:, pad_id+1] .= 0
        return weight
    end
    pos_weight = getweight(weight_init(max_pos, dims, factor), Layers.Embed,
                           state_dict, joinname(prefix, "position_embeddings.weight"))
    segment_weight = getweight(weight_init(max_pos, dims, factor), Layers.Embed,
                               state_dict, joinname(prefix, "token_type_embeddings.weight"))
    embed = CompositeEmbedding(
        token = Layers.Embed(token_weight),
        position = Layers.FixedLenPositionEmbed(pos_weight),
        segment = (.+, Layers.Embed(segment_weight), bert_ones_like)
    )
    ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(prefix, "LayerNorm"))
    return Layers.Chain(embed, Layers.DropoutLayer(ln, p))
end

function load_model(_type::Type{<:HGFBertPreTrainedModel}, ::Type{<:SelfAttention}, cfg, state_dict, prefix)
    head, dims = cfg[:num_attention_heads], cfg[:hidden_size]
    @assert dims % head == 0 "The hidden size is not a multiple of the number of attention heads."
    p = cfg[:attention_probs_dropout_prob]; p = iszero(p) ? nothing : p
    pe_type = cfg[:position_embedding_type]
    pe_type == "absolute" || load_error("Right now only absolute PE is supported in Bert.")
    return_score = cfg[:output_attentions]
    factor = Float32(cfg[:initializer_range])
    w_init = weight_init(dims, dims, factor)
    b_init = zero_init(dims)
    q_weight = getweight(w_init, Array, state_dict, joinname(prefix, "self.query.weight"))
    k_weight = getweight(w_init, Array, state_dict, joinname(prefix, "self.key.weight"))
    v_weight = getweight(w_init, Array, state_dict, joinname(prefix, "self.value.weight"))
    q_bias = getweight(b_init, Array, state_dict, joinname(prefix, "self.query.bias"))
    k_bias = getweight(b_init, Array, state_dict, joinname(prefix, "self.key.bias"))
    v_bias = getweight(b_init, Array, state_dict, joinname(prefix, "self.value.bias"))
    qkv_proj = Layers.Fork(
        Layers.Dense(q_weight, q_bias),
        Layers.Dense(k_weight, k_bias),
        Layers.Dense(v_weight, v_bias))
    o_weight = getweight(w_init, Array, state_dict, joinname(prefix, "output.dense.weight"))
    o_bias = getweight(b_init, Array, state_dict, joinname(prefix, "output.dense.bias"))
    o_proj = Layers.Dense(o_weight, o_bias)
    op = MultiheadQKVAttenOp(head, p)
    return_score && (op = WithScore(op))
    return SelfAttention(op, qkv_proj, o_proj)
end

function load_model(
    _type::Type{<:HGFBertPreTrainedModel}, ::Type{<:Layers.Chain{<:Tuple{Layers.Dense, Layers.Dense}}},
    cfg, state_dict, prefix
)
    dims, ff_dims = cfg[:hidden_size], cfg[:intermediate_size]
    factor = Float32(cfg[:initializer_range])
    p = cfg[:hidden_dropout_prob]; p = iszero(p) ? nothing : p
    act = ACT2FN[Symbol(cfg[:hidden_act])]
    wi_weight = getweight(weight_init(dims, ff_dims, factor), Array,
                          state_dict, joinname(prefix, "intermediate.dense.weight"))
    wi_bias = getweight(zero_init(ff_dims), Array, state_dict, joinname(prefix, "intermediate.dense.bias"))
    wo_weight = getweight(weight_init(ff_dims, dims, factor), Array,
                          state_dict, joinname(prefix, "output.dense.weight"))
    wo_bias = getweight(zero_init(dims), Array, state_dict, joinname(prefix, "output.dense.bias"))
    return Layers.Chain(Layers.Dense(act, wi_weight, wi_bias), Layers.Dense(wo_weight, wo_bias))
end

function load_model(_type::Type{<:HGFBertPreTrainedModel}, ::Type{<:TransformerBlock}, cfg, state_dict, prefix)
    n = cfg[:num_hidden_layers]
    p = cfg[:hidden_dropout_prob]; p = iszero(p) ? nothing : p
    collect_output = cfg[:output_attentions] || cfg[:output_hidden_states]
    cfg[:add_cross_attention] && load_error("Decoder Bert is not support.")
    blocks = []
    for i = 1:n
        lprefix = joinname(prefix, :layer, i-1)
        sa = load_model(_type, SelfAttention, cfg, state_dict, joinname(lprefix, "attention"))
        sa_ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(lprefix, "attention.output.LayerNorm"))
        sa = Layers.PostNormResidual(Layers.DropoutLayer(sa, p), sa_ln)
        ff = load_model(_type, Layers.Chain{Tuple{Layers.Dense, Layers.Dense}}, cfg, state_dict, lprefix)
        ff_ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(lprefix, "output.LayerNorm"))
        ff = Layers.PostNormResidual(Layers.DropoutLayer(ff, p), ff_ln)
        block = TransformerBlock(sa, ff)
        push!(blocks, block)
    end
    collect_f = collect_output ? Layers.collect_outputs : nothing
    trf = Transformer(Tuple(blocks), collect_f)
    return trf
end

function get_state_dict(m::HGFBertModel, state_dict, prefix)
    get_state_dict(HGFBertModel, m.embed[1], state_dict, joinname(prefix, "embeddings"))
    get_state_dict(HGFBertModel, m.embed[2], state_dict, joinname(prefix, "embeddings.LayerNorm"))
    get_state_dict(HGFBertModel, m.encoder, state_dict, joinname(prefix, "encoder"))
    if !isnothing(m.pooler)
        get_state_dict(HGFBertModel, m.pooler.layer.dense, state_dict, joinname(prefix, "pooler.dense"))
    end
    return state_dict
end

function get_state_dict(m::HGFBertForPreTraining, state_dict, prefix)
    get_state_dict(m.model, state_dict, joinname(prefix, "bert"))
    get_state_dict(HGFBertModel, m.cls[1].layer[1],
                   state_dict, joinname(prefix, "cls.predictions.transform.dense"))
    get_state_dict(HGFBertModel, m.cls[1].layer[2], state_dict, joinname(prefix, "cls.predictions.transform.LayerNorm"))
    get_state_dict(HGFBertModel, m.cls[1].layer[3], state_dict, joinname(prefix, "cls.predictions"))
    get_state_dict(HGFBertModel, m.cls[2], state_dict, joinname(prefix, "cls.seq_relationship"))
    return state_dict
end

function get_state_dict(m::Union{HGFBertLMHeadModel, HGFBertForMaskedLM}, state_dict, prefix)
    get_state_dict(m.model, state_dict, joinname(prefix, "bert"))
    get_state_dict(HGFBertModel, m.cls.layer[1],
                   state_dict, joinname(prefix, "cls.predictions.transform.dense"))
    get_state_dict(HGFBertModel, m.cls.layer[2], state_dict, joinname(prefix, "cls.predictions.transform.LayerNorm"))
    get_state_dict(HGFBertModel, m.cls.layer[3], state_dict, joinname(prefix, "cls.predictions"))
    return state_dict
end

function get_state_dict(m::HGFBertForNextSentencePrediction, state_dict, prefix)
    get_state_dict(m.model, state_dict, joinname(prefix, "bert"))
    get_state_dict(HGFBertModel, m.cls.layer.layer, state_dict, joinname(prefix, "cls.seq_relationship"))
    return state_dict
end

function get_state_dict(m::Union{HGFBertForSequenceClassification, HGFBertForTokenClassification}, state_dict, prefix)
    get_state_dict(m.model, state_dict, joinname(prefix, "bert"))
    get_state_dict(HGFBertModel, m.cls, state_dict, joinname(prefix, "classifier"))
    return state_dict
end

function get_state_dict(m::HGFBertForQuestionAnswering, state_dict, prefix)
    get_state_dict(m.model, state_dict, joinname(prefix, "bert"))
    get_state_dict(HGFBertModel, m.cls.dense, state_dict, joinname(prefix, "qa_outputs"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFBertPreTrainedModel}, m::Layers.EmbedDecoder, state_dict, prefix)
    get_state_dict(p, m.embed, state_dict, joinname(prefix, "decoder"))
    state_dict[joinname(prefix, "bias")] = m.bias
    return state_dict
end

function get_state_dict(p::Type{<:HGFBertPreTrainedModel}, m::CompositeEmbedding, state_dict, prefix)
    get_state_dict(p, m.token, state_dict, joinname(prefix, "word_embeddings"))
    get_state_dict(p, m.position.embed, state_dict, joinname(prefix, "position_embeddings"))
    get_state_dict(p, m.segment.embed, state_dict, joinname(prefix, "token_type_embeddings"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFBertPreTrainedModel}, m::SelfAttention, state_dict, prefix)
    get_state_dict(p, m.qkv_proj.layers[1], state_dict, joinname(prefix, "self.query"))
    get_state_dict(p, m.qkv_proj.layers[2], state_dict, joinname(prefix, "self.key"))
    get_state_dict(p, m.qkv_proj.layers[3], state_dict, joinname(prefix, "self.value"))
    get_state_dict(p, m.o_proj, state_dict, joinname(prefix, "output.dense"))
    return state_dict
end

function get_state_dict(
    p::Type{<:HGFBertPreTrainedModel}, m::Layers.Chain{<:Tuple{Layers.Dense, Layers.Dense}},
    state_dict, prefix
)
    get_state_dict(p, m[1], state_dict, joinname(prefix, "intermediate.dense"))
    get_state_dict(p, m[2], state_dict, joinname(prefix, "output.dense"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFBertPreTrainedModel}, m::TransformerBlock, state_dict, prefix)
    get_state_dict(p, m.attention.layer, state_dict, joinname(prefix, "attention"))
    get_state_dict(p, m.attention.norm, state_dict, joinname(prefix, "attention.output.LayerNorm"))
    get_state_dict(p, m.feedforward.layer, state_dict, prefix)
    get_state_dict(p, m.feedforward.norm, state_dict, joinname(prefix, "output.LayerNorm"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFBertPreTrainedModel}, m::Transformer, state_dict, prefix)
    for (i, t) in enumerate(m.blocks)
        get_state_dict(p, t, state_dict, joinname(prefix, :layer, i-1))
    end
    return state_dict
end
