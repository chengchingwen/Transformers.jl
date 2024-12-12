using ..Layers
using ..Layers: CompositeEmbedding, SelfAttention, MultiheadQKVAttenOp
using ChainRulesCore
using Functors
using FillArrays
using NNlib
using Static

using NeuralAttentionlib
using NeuralAttentionlib: WithScore

struct DistilBertQA{D}
	dense::D
end
@functor DistilBertQA
@fluxlayershow DistilBertQA

function (m::DistilBertQA)(x)
	logits = m.dense(x)
	start_logit = _slice(logits, 1)
	end_logit = _slice(logits, 2)
	return (; start_logit, end_logit)
end
(m::DistilBertQA)(nt::NamedTuple) = merge(nt, m(nt.hidden_state))

@hgfdef DistilBert (
	Model => begin
		outputs = model.encoder(model.embed(nt))
	end,
	# ForPreTraining,
	ForMaskedLM,
	# ForSequenceClassification,
	# ForTokenClassification,
	# ForQuestionAnswering,
	# ForMultipleChoice,
)

basemodelkey(::Type{<:HGFPreTrained{:distilbert}}) = :distilbert

distilbert_ones_like(x::AbstractArray) = Ones{Int}(Base.tail(size(x)))
ChainRulesCore.@non_differentiable distilbert_ones_like(x)

load_model(_type::Type{HGFDistilBertModel}, cfg, state_dict, prefix) =
	load_model(_type, _type, cfg, state_dict, prefix)

function load_model(_type::Type, ::Type{HGFDistilBertModel}, cfg, state_dict, prefix)
	embed = load_model(_type, CompositeEmbedding, cfg, state_dict, joinname(prefix, "embeddings"))
	encoder = load_model(_type, TransformerBlock, cfg, state_dict, joinname(prefix, "transformer"))
	dims = cfg[:hidden_size]
	factor = Float32(cfg[:initializer_range])
	return HGFDistilBertModel(embed, encoder)
end

function load_model(
	_type::Type{<:Union{
		HGFDistilBertForMaskedLM,
		# HGFDistilBertForTokenClassification,HGFDistilBertForQuestionAnswering,
	}},
	::Type{HGFDistilBertModel}, cfg, state_dict, prefix)
	embed = load_model(_type, CompositeEmbedding, cfg, state_dict, joinname(prefix, "embeddings"))
	encoder = load_model(_type, TransformerBlock, cfg, state_dict, joinname(prefix, "transformer"))
	return HGFDistilBertModel(embed, encoder)
end

# function load_model(_type::Type{HGFDistilBertForPreTraining}, cfg, state_dict, prefix)
#     distilbert = load_model(_type, HGFDistilBertForMaskedLM, cfg, state_dict, prefix)
#     model, lmhead = distilbert.model, distilbert.cls
#     seqhead = load_model(HGFDistilBertForNextSentencePrediction, Layers.Dense, cfg, state_dict, prefix)
#     cls = Layers.Chain(lmhead, seqhead)
#     return HGFDistilBertForPreTraining(model, cls)
# end

function load_model(_type::Type{<:Union{HGFDistilBertForMaskedLM}}, cfg, state_dict, prefix)
	model = load_model(_type, HGFDistilBertModel, cfg, state_dict, joinname(prefix, "distilbert"))
	dims, vocab_size, pad_id = cfg[:hidden_size], cfg[:vocab_size], cfg[:pad_token_id]
	factor = Float32(cfg[:initializer_range])
	act = ACT2FN[Symbol(cfg[:hidden_act])]
	# HGFDistilBertPredictionHeadTransform
	head_weight = getweight(weight_init(dims, dims, factor), Array,
		state_dict, joinname(prefix, "vocab_transform.weight"))
	head_bias = getweight(zero_init(dims), Array,
		state_dict, joinname(prefix, "vocab_transform.bias"))
	head_ln = load_model(HGFDistilBertModel, Layers.LayerNorm, cfg,
		state_dict, joinname(prefix, "vocab_layer_norm"))
	# HGFDistilBertLMPredictionHead
	if cfg[:tie_word_embeddings]
		embedding = model.embed[1].token.embeddings
	else
		embedding = getweight(Layers.Embed, state_dict, joinname(prefix, "vocab_projector.weight")) do
			weight = weight_init(vocab_size, dims, factor)()
			weight[:, pad_id+1] .= 0
			return weight
		end
	end
	bias = getweight(zero_init(vocab_size), Array, state_dict, joinname(prefix, "vocab_projector.bias"))
	lmhead = Layers.Chain(Layers.Dense(act, head_weight, head_bias), head_ln, Layers.EmbedDecoder(Layers.Embed(embedding), bias))
	return _type(model, Layers.Branch{(:logit,), (:hidden_state,)}(lmhead))
end

# function load_model(_type::Type{HGFDistilBertForSequenceClassification}, cfg, state_dict, prefix)
#     model = load_model(HGFDistilBertModel, cfg, state_dict, joinname(prefix, "distilbert"))
#     dims, nlabel = cfg[:hidden_size], cfg[:num_labels]
#     factor = Float32(cfg[:initializer_range])
#     weight = getweight(weight_init(dims, nlabel, factor), Array, state_dict, joinname(prefix, "classifier.weight"))
#     bias = getweight(zero_init(nlabel), Array, state_dict, joinname(prefix, "classifier.bias"))
#     cls = Layers.Branch{(:logit,),(:pooled,)}(Layers.Dense(weight, bias))
#     return HGFDistilBertForSequenceClassification(model, cls)
# end

# function load_model(
#     _type::Type{HGFDistilBertForMultipleChoice}, cfg,
#     state_dict = OrderedDict{String, Any}(), prefix = ""
# )
#     model = load_model(HGFDistilBertModel, cfg, state_dict, joinname(prefix, "distilbert"))

# end

# function load_model(_type::Type{HGFDistilBertForTokenClassification}, cfg, state_dict, prefix)
#     model = load_model(_type, HGFDistilBertModel, cfg, state_dict, joinname(prefix, "distilbert"))
#     dims, nlabel = cfg[:hidden_size], cfg[:num_labels]
#     factor = Float32(cfg[:initializer_range])
#     weight = getweight(weight_init(dims, nlabel, factor), Array, state_dict, joinname(prefix, "classifier.weight"))
#     bias = getweight(zero_init(nlabel), Array, state_dict, joinname(prefix, "classifier.bias"))
#     cls = Layers.Branch{(:logit,),(:hidden_state,)}(Layers.Dense(weight, bias))
#     return HGFDistilBertForTokenClassification(model, cls)
# end

# function load_model(_type::Type{HGFDistilBertForQuestionAnswering}, cfg, state_dict, prefix)
#     model = load_model(_type, HGFDistilBertModel, cfg, state_dict, joinname(prefix, "distilbert"))
#     dims, nlabel = cfg[:hidden_size], cfg[:num_labels]
#     factor = Float32(cfg[:initializer_range])
#     weight = getweight(weight_init(dims, nlabel, factor), Array,
#         state_dict, joinname(prefix, "qa_outputs.weight"))
#     bias = getweight(zero_init(nlabel), Array, state_dict, joinname(prefix, "qa_outputs.bias"))
#     cls = DistilBertQA(Layers.Dense(weight, bias))
#     return HGFDistilBertForQuestionAnswering(model, cls)
# end

function load_model(_type::Type{<:HGFDistilBertPreTrainedModel}, ::Type{<:Layers.LayerNorm}, cfg, state_dict, prefix)
	dims = cfg[:hidden_size]
	ln_ϵ = Float32(cfg[:layer_norm_eps])
	weight_name = joinname(prefix, "weight")
	bias_name = joinname(prefix, "bias")
	ln_weight = getweight(one_init(dims), Array, state_dict, weight_name)
	ln_bias = getweight(zero_init(dims), Array, state_dict, bias_name)
	return Layers.LayerNorm(ln_weight, ln_bias, ln_ϵ)
end

function load_model(_type::Type{<:HGFDistilBertPreTrainedModel}, ::Type{<:CompositeEmbedding}, cfg, state_dict, prefix)
	vocab_size, dims, pad_id = cfg[:vocab_size], cfg[:hidden_size], cfg[:pad_token_id]
	max_pos, n_type = cfg[:max_position_embeddings], cfg[:type_vocab_size]
	p = cfg[:hidden_dropout_prob]
	p = iszero(p) ? nothing : p
	pe_type = cfg[:position_embedding_type]
	pe_type == "absolute" || load_error("Right now only absolute PE is supported in DistilBert.")
	factor = Float32(cfg[:initializer_range])
	token_weight = getweight(Layers.Embed, state_dict, joinname(prefix, "word_embeddings.weight")) do
		weight = weight_init(vocab_size, dims, factor)()
		weight[:, pad_id+1] .= 0
		return weight
	end
	pos_weight = getweight(weight_init(max_pos, dims, factor), Layers.Embed,
		state_dict, joinname(prefix, "position_embeddings.weight"))
	embed = CompositeEmbedding(
		token = Layers.Embed(token_weight),
		position = Layers.FixedLenPositionEmbed(pos_weight),
	)
	ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(prefix, "LayerNorm"))
	return Layers.Chain(embed, Layers.DropoutLayer(ln, p))
end

function load_model(_type::Type{<:HGFDistilBertPreTrainedModel}, ::Type{<:SelfAttention}, cfg, state_dict, prefix)
	head, dims = cfg[:num_attention_heads], cfg[:hidden_size]
	@assert dims % head == 0 "The hidden size is not a multiple of the number of attention heads."
	p = cfg[:attention_probs_dropout_prob]
	p = iszero(p) ? nothing : p
	pe_type = cfg[:position_embedding_type]
	pe_type == "absolute" || load_error("Right now only absolute PE is supported in DistilBert.")
	return_score = cfg[:output_attentions]
	factor = Float32(cfg[:initializer_range])
	w_init = weight_init(dims, dims, factor)
	b_init = zero_init(dims)
	q_weight = getweight(w_init, Array, state_dict, joinname(prefix, "q_lin.weight"))
	k_weight = getweight(w_init, Array, state_dict, joinname(prefix, "k_lin.weight"))
	v_weight = getweight(w_init, Array, state_dict, joinname(prefix, "v_lin.weight"))
	q_bias = getweight(b_init, Array, state_dict, joinname(prefix, "q_lin.bias"))
	k_bias = getweight(b_init, Array, state_dict, joinname(prefix, "k_lin.bias"))
	v_bias = getweight(b_init, Array, state_dict, joinname(prefix, "v_lin.bias"))
	qkv_proj = Layers.Fork(
		Layers.Dense(q_weight, q_bias),
		Layers.Dense(k_weight, k_bias),
		Layers.Dense(v_weight, v_bias))
	o_weight = getweight(w_init, Array, state_dict, joinname(prefix, "out_lin.weight"))
	o_bias = getweight(b_init, Array, state_dict, joinname(prefix, "out_lin.bias"))
	o_proj = Layers.Dense(o_weight, o_bias)
	op = MultiheadQKVAttenOp(head, p)
	return_score && (op = WithScore(op))
	return SelfAttention(op, qkv_proj, o_proj)
end

function load_model(
	_type::Type{<:HGFDistilBertPreTrainedModel}, ::Type{<:Layers.Chain{<:Tuple{Layers.Dense, Layers.Dense}}},
	cfg, state_dict, prefix,
)
	dims, ff_dims = cfg[:hidden_size], cfg[:intermediate_size]
	factor = Float32(cfg[:initializer_range])
	p = cfg[:hidden_dropout_prob]
	p = iszero(p) ? nothing : p
	act = ACT2FN[Symbol(cfg[:hidden_act])]
	wi_weight = getweight(weight_init(dims, ff_dims, factor), Array,
		state_dict, joinname(prefix, "ffn.lin1.weight"))
	wi_bias = getweight(zero_init(ff_dims), Array, state_dict, joinname(prefix, "ffn.lin1.bias"))

	wo_weight = getweight(weight_init(ff_dims, dims, factor), Array,
		state_dict, joinname(prefix, "ffn.lin2.weight"))
	wo_bias = getweight(zero_init(dims), Array, state_dict, joinname(prefix, "ffn.lin2.bias"))
	return Layers.Chain(Layers.Dense(act, wi_weight, wi_bias), Layers.Dense(wo_weight, wo_bias))
end

function load_model(_type::Type{<:HGFDistilBertPreTrainedModel}, ::Type{<:TransformerBlock}, cfg, state_dict, prefix)
	n = cfg[:num_hidden_layers]
	p = cfg[:hidden_dropout_prob]
	p = iszero(p) ? nothing : p
	collect_output = cfg[:output_attentions] || cfg[:output_hidden_states]
	cfg[:add_cross_attention] && load_error("Decoder DistilBert is not support.")
	blocks = []
	for i ∈ 1:n
		lprefix = joinname(prefix, :layer, i - 1)
		sa = load_model(_type, SelfAttention, cfg, state_dict, joinname(lprefix, "attention"))
		sa_ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(lprefix, "sa_layer_norm"))
		sa = Layers.PostNormResidual(Layers.DropoutLayer(sa, p), sa_ln)
		ff = load_model(_type, Layers.Chain{Tuple{Layers.Dense, Layers.Dense}}, cfg, state_dict, lprefix)
		ff_ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(lprefix, "output_layer_norm"))
		ff = Layers.PostNormResidual(Layers.DropoutLayer(ff, p), ff_ln)
		block = TransformerBlock(sa, ff)
		push!(blocks, block)
	end
	collect_f = collect_output ? Layers.collect_outputs : nothing
	trf = Transformer(Tuple(blocks), collect_f)
	return trf
end

function get_state_dict(m::HGFDistilBertModel, state_dict, prefix)
	get_state_dict(HGFDistilBertModel, m.embed[1], state_dict, joinname(prefix, "embeddings"))
	get_state_dict(HGFDistilBertModel, m.embed[2], state_dict, joinname(prefix, "embeddings.LayerNorm"))
	get_state_dict(HGFDistilBertModel, m.encoder, state_dict, joinname(prefix, "transformer"))
	return state_dict
end

# function get_state_dict(m::HGFDistilBertForPreTraining, state_dict, prefix)
#     get_state_dict(m.model, state_dict, joinname(prefix, "distilbert"))
#     get_state_dict(HGFDistilBertModel, m.cls[1].layer[1],
#         state_dict, joinname(prefix, "vocab_transform"))
#     get_state_dict(HGFDistilBertModel, m.cls[1].layer[2], state_dict, joinname(prefix, "vocab_layer_norm"))
#     get_state_dict(HGFDistilBertModel, m.cls[1].layer[3], state_dict, joinname(prefix, "vocab_projector"))
#     get_state_dict(HGFDistilBertModel, m.cls[2], state_dict, joinname(prefix, "cls.seq_relationship"))
#     return state_dict
# end

function get_state_dict(m::Union{HGFDistilBertForMaskedLM}, state_dict, prefix)
	get_state_dict(m.model, state_dict, joinname(prefix, "distilbert"))
	get_state_dict(HGFDistilBertModel, m.cls.layer[1], state_dict, joinname(prefix, "vocab_transform"))
	get_state_dict(HGFDistilBertModel, m.cls.layer[2], state_dict, joinname(prefix, "vocab_layer_norm"))
	get_state_dict(HGFDistilBertModel, m.cls.layer[3], state_dict, joinname(prefix, "vocab_projector"))
	return state_dict
end

# function get_state_dict(m::Union{HGFDistilBertForSequenceClassification,HGFDistilBertForTokenClassification}, state_dict, prefix)
#     get_state_dict(m.model, state_dict, joinname(prefix, "distilbert"))
#     get_state_dict(HGFDistilBertModel, m.cls, state_dict, joinname(prefix, "classifier"))
#     return state_dict
# end

# function get_state_dict(m::HGFDistilBertForQuestionAnswering, state_dict, prefix)
#     get_state_dict(m.model, state_dict, joinname(prefix, "distilbert"))
#     get_state_dict(HGFDistilBertModel, m.cls.dense, state_dict, joinname(prefix, "qa_outputs"))
#     return state_dict
# end

function get_state_dict(p::Type{<:HGFDistilBertPreTrainedModel}, m::Layers.EmbedDecoder, state_dict, prefix)
	get_state_dict(p, m.embed, state_dict, joinname(prefix, "decoder"))
	state_dict[joinname(prefix, "bias")] = m.bias
	return state_dict
end

function get_state_dict(p::Type{<:HGFDistilBertPreTrainedModel}, m::CompositeEmbedding, state_dict, prefix)
	get_state_dict(p, m.token, state_dict, joinname(prefix, "word_embeddings"))
	get_state_dict(p, m.position.embed, state_dict, joinname(prefix, "position_embeddings"))
	return state_dict
end

function get_state_dict(p::Type{<:HGFDistilBertPreTrainedModel}, m::SelfAttention, state_dict, prefix)
	get_state_dict(p, m.qkv_proj.layers[1], state_dict, joinname(prefix, "q_lin"))
	get_state_dict(p, m.qkv_proj.layers[2], state_dict, joinname(prefix, "k_lin"))
	get_state_dict(p, m.qkv_proj.layers[3], state_dict, joinname(prefix, "v_lin"))
	get_state_dict(p, m.o_proj, state_dict, joinname(prefix, "out_lin"))
	return state_dict
end

function get_state_dict(
	p::Type{<:HGFDistilBertPreTrainedModel}, m::Layers.Chain{<:Tuple{Layers.Dense, Layers.Dense}},
	state_dict, prefix,
)
	get_state_dict(p, m[1], state_dict, joinname(prefix, "lin1"))
	get_state_dict(p, m[2], state_dict, joinname(prefix, "lin2"))
	return state_dict
end

function get_state_dict(p::Type{<:HGFDistilBertPreTrainedModel}, m::TransformerBlock, state_dict, prefix)
	get_state_dict(p, m.attention.layer, state_dict, joinname(prefix, "attention"))
	get_state_dict(p, m.attention.norm, state_dict, joinname(prefix, "sa_layer_norm"))
	get_state_dict(p, m.feedforward.layer, state_dict, joinname(prefix, "ffn"))
	get_state_dict(p, m.feedforward.norm, state_dict, joinname(prefix, "output_layer_norm"))
	return state_dict
end

function get_state_dict(p::Type{<:HGFDistilBertPreTrainedModel}, m::Transformer, state_dict, prefix)
	for (i, t) in enumerate(m.blocks)
		get_state_dict(p, t, state_dict, joinname(prefix, :layer, i - 1))
	end
	return state_dict
end
