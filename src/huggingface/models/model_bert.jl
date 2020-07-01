# Bert specific initializers

function FakeTHLinear(config::HGFBertConfig, hi, ho; bias=true)
  weight = randn(Float32, ho, hi) .* config.initializer_range
  if bias
    bias = zeros(Float32, ho)
  else
    bias = nothing
  end
  FakeTHLinear(weight, bias)
end

function FakeTHEmbedding(config::HGFBertConfig, num, dims; pad_idx=nothing)
  weight = randn(Float32, dims, num)
  !isnothing(pad_idx) && (@view(weight[:, pad_idx+1]) .= 0)
  FakeTHEmbedding(pad_idx, weight)
end

function FakeTHLayerNorm(config::HGFBertConfig, dims; eps::Float32=1e-05)
  weight = ones(Float32, dims)
  bias = zeros(Float32, dims)
  FakeTHLayerNorm(eps, weight, bias)
end

# HGF Bert compounts

# embedding

struct HGFBertEmbeddings <: THModule
  LayerNorm::FakeTHLayerNorm
  word_embeddings::FakeTHEmbedding
  position_embeddings::FakeTHEmbedding
  token_type_embeddings::FakeTHEmbedding
end

@functor HGFBertEmbeddings

function HGFBertEmbeddings(config::HGFBertConfig)
  layernorm = FakeTHLayerNorm(config, config.hidden_size; eps=config.layer_norm_eps)

  word_emb = FakeTHEmbedding(config, config.vocab_size,
                             config.hidden_size;
                             pad_idx=config.pad_token_id)
  posi_emb = FakeTHEmbedding(config, config.max_position_embeddings,
                             config.hidden_size)
  toke_emb = FakeTHEmbedding(config, config.type_vocab_size,
                             config.hidden_size)

  HGFBertEmbeddings(layernorm, word_emb, posi_emb, toke_emb)
end

# self attention part

struct HGFBertSelfAttention <: THModule
  query::FakeTHLinear
  key::FakeTHLinear
  value::FakeTHLinear
end

@functor HGFBertSelfAttention

function HGFBertSelfAttention(config::HGFBertConfig)
  attention_head_size = config.hidden_size ÷ config.num_attention_heads
  all_head_size = config.num_attention_heads * attention_head_size

  query = FakeTHLinear(config, config.hidden_size, all_head_size)
  key   = FakeTHLinear(config, config.hidden_size, all_head_size)
  value = FakeTHLinear(config, config.hidden_size, all_head_size)

  HGFBertSelfAttention(query, key, value)
end

# self attention output part

struct HGFBertSelfOutput <: THModule
  LayerNorm::FakeTHLayerNorm
  dense::FakeTHLinear
end

@functor HGFBertSelfOutput

function HGFBertSelfOutput(config::HGFBertConfig)
  layernorm = FakeTHLayerNorm(config, config.hidden_size; eps=config.layer_norm_eps)
  dense = FakeTHLinear(config, config.hidden_size, config.hidden_size)
  HGFBertSelfOutput(layernorm, dense)
end

# self attention

struct HGFBertAttention <: THModule
  self::HGFBertSelfAttention
  output::HGFBertSelfOutput
end

@functor HGFBertAttention

function HGFBertAttention(config::HGFBertConfig)
  self = HGFBertSelfAttention(config)
  output = HGFBertSelfOutput(config)
  HGFBertAttention(self, output)
end

# positionwise first dense

struct HGFBertIntermediate{F} <: THModule
  intermediate_act::F
  dense::FakeTHLinear
end

@functor HGFBertIntermediate (dense,)

function HGFBertIntermediate(config::HGFBertConfig)
  global ACT2FN
  act = ACT2FN[Symbol(config.hidden_act)]
  dense = FakeTHLinear(config, config.hidden_size, config.intermediate_size)
  HGFBertIntermediate(act, dense)
end

# positionwise second dense

struct HGFBertOutput <: THModule
  dense::FakeTHLinear
  LayerNorm::FakeTHLayerNorm
end

@functor HGFBertOutput

function HGFBertOutput(config::HGFBertConfig)
  dense = FakeTHLinear(config, config.intermediate_size, config.hidden_size)
  layernorm = FakeTHLayerNorm(config, config.hidden_size; eps=config.layer_norm_eps)
  HGFBertOutput(dense, layernorm)
end

# transformer layer

struct HGFBertLayer{DEC<:Union{Nothing, HGFBertAttention}} <: THModule
  attention::HGFBertAttention
  crossattention::DEC
  intermediate::HGFBertIntermediate
  output::HGFBertOutput
end

HGFBertLayer(a, i, o) = HGFBertLayer(a, nothing, i, o)

_is_decode(::HGFBertLayer{Nothing}) = false
_is_decode(::HGFBertLayer) = true

Functors.functor(::Type{<:HGFBertLayer}, layer) = (_is_decode(layer) ?
    (attention = layer.attention, intermediate = layer.intermediate, output = layer.output) :
    (attention = layer.attention, crossattention = layer.crossattention, intermediate = layer.intermediate, output = layer.output)),
    y ->HGFBertLayer(y...)

function HGFBertLayer(config::HGFBertConfig)
  attention = HGFBertAttention(config)
  crossattention = config.is_decode ?
    HGFBertAttention(config) :
    nothing
  intermediate = HGFBertIntermediate(config)
  output = HGFBertOutput(config)
  HGFBertLayer(attention, crossattention, intermediate, output)
end

# stacked transformers

struct HGFBertEncoder <: THModule
  layer::FakeTHModuleList
end

@functor HGFBertEncoder

function HGFBertEncoder(config::HGFBertConfig)
  layer = FakeTHModuleList(
    [HGFBertLayer(config) for _ in 1:config.num_hidden_layers]
  )
  HGFBertEncoder(layer)
end

# classify token projection

struct HGFBertPooler <: THModule
  dense::FakeTHLinear
end

@functor HGFBertPooler

function HGFBertPooler(config::HGFBertConfig)
  dense = FakeTHLinear(config, config.hidden_size, config.hidden_size)
  HGFBertPooler(dense)
end

# label prediction layer

struct HGFBertPredictionHeadTransform{F} <: THModule
  transform_act_fn::F
  dense::FakeTHLinear
  LayerNorm::FakeTHLayerNorm
end

@functor HGFBertPredictionHeadTransform

function HGFBertPredictionHeadTransform(config::HGFBertConfig)
  global ACT2FN
  dense = FakeTHLinear(config, config.hidden_size, config.hidden_size)
  act = ACT2FN[Symbol(config.hidden_act)]
  layernorm = FakeTHLayerNorm(config, config.hidden_size; eps=config.layer_norm_eps)

  return HGFBertPredictionHeadTransform(act, dense, layernorm)
end

# language model prediction layer

struct HGFBertLMPredictionHead{B<:AbstractArray} <: THModule
  transform::HGFBertPredictionHeadTransform
  decoder::FakeTHLinear
  bias::B
end

@functor HGFBertLMPredictionHead

function HGFBertLMPredictionHead(config::HGFBertConfig; input_embedding=nothing)
  trans = HGFBertPredictionHeadTransform(config)
  if isnothing(input_embedding)
    decoder = FakeTHLinear(config, config.hidden_size, config.vocab_size; bias=false)
  else
    decoder = FakeTHLinear(Transpose(input_embedding), nothing)
  end
  bias = zeros(Float32, config.vocab_size)
  return HGFBertLMPredictionHead(trans, decoder, bias)
end

# language model prediction wrapper

struct HGFBertOnlyMLMHead <: THModule
  predictions::HGFBertLMPredictionHead
end

@functor HGFBertOnlyMLMHead

function HGFBertOnlyMLMHead(config::HGFBertConfig; input_embedding=nothing)
  predictions = HGFBertLMPredictionHead(config; input_embedding=input_embedding)
  return HGFBertOnlyMLMHead(predictions)
end

# next sentence prediction layer

struct HGFBertOnlyNSPHead <: THModule
  seq_relationship::FakeTHLinear
end

@functor HGFBertOnlyNSPHead

function HGFBertOnlyNSPHead(config::HGFBertConfig)
  seq_relationship = FakeTHLinear(config, config.hidden_size, 2)
  return HGFBertOnlyNSPHead(seq_relationship)
end

# pretrain prediction layers

struct HGFBertPreTrainingHeads <: THModule
  predictions::HGFBertLMPredictionHead
  seq_relationship::FakeTHLinear
end

@functor HGFBertPreTrainingHeads

function HGFBertPreTrainingHeads(config::HGFBertConfig; input_embedding=nothing)
  predictions = HGFBertLMPredictionHead(config; input_embedding=input_embedding)
  seq_relationship = FakeTHLinear(config, config.hidden_size, 2)
  return HGFBertPreTrainingHeads(predictions, seq_relationship)
end

# bert model without prediction
abstract type HGFBertPreTrainedModel <: HGFPreTrainedModel end

struct HGFBertModel <: HGFBertPreTrainedModel
  embeddings::HGFBertEmbeddings
  encoder::HGFBertEncoder
  pooler::HGFBertPooler
end

@functor HGFBertModel

function HGFBertModel(config::HGFBertConfig)
  embeddings = HGFBertEmbeddings(config)
  encoder = HGFBertEncoder(config)
  pooler = HGFBertPooler(config)
  HGFBertModel(embeddings, encoder, pooler)
end

get_input_embedding(model::HGFBertModel) = model.embeddings.word_embeddings.weight

# bert models for different task

# pretrain

struct HGFBertForPreTraining <: HGFBertPreTrainedModel
  bert::HGFBertModel
  cls::HGFBertPreTrainingHeads
end

@functor HGFBertForPreTraining

function HGFBertForPreTraining(config::HGFBertConfig)
  bert = HGFBertModel(config)
  input_embedding = get_input_embedding(bert)
  cls = HGFBertPreTrainingHeads(config; input_embedding=input_embedding)
  return HGFBertForPreTraining(bert, cls)
end

# clm finetune

struct HGFBertLMHeadModel <: HGFBertPreTrainedModel
  bert::HGFBertModel
  cls::HGFBertOnlyMLMHead
end

@functor HGFBertLMHeadModel

function HGFBertLMHeadModel(config::HGFBertConfig)
  bert = HGFBertModel(config)
  input_embedding = get_input_embedding(bert)
  cls = HGFBertOnlyMLMHead(config; input_embedding=input_embedding)
  return HGFBertLMHeadModel(bert, cls)
end

# maked lm

struct HGFBertForMaskedLM <: HGFBertPreTrainedModel
  bert::HGFBertModel
  cls::HGFBertOnlyMLMHead
end

@functor HGFBertForMaskedLM

function HGFBertForMaskedLM(config::HGFBertConfig)
  bert = HGFBertModel(config)
  input_embedding = get_input_embedding(bert)
  cls = HGFBertOnlyMLMHead(config; input_embedding=input_embedding)
  return HGFBertForMaskedLM(bert, cls)
end

# next sentence

struct HGFBertForNextSentencePrediction <: HGFBertPreTrainedModel
  bert::HGFBertModel
  cls::HGFBertOnlyNSPHead
end

@functor HGFBertForNextSentencePrediction

function HGFBertForNextSentencePrediction(config::HGFBertConfig)
  bert = HGFBertModel(config)
  cls = HGFBertOnlyNSPHead(config)
  return HGFBertForNextSentencePrediction(bert, cls)
end

# seq classify

struct HGFBertForSequenceClassification <: HGFBertPreTrainedModel
  bert::HGFBertModel
  classifier::FakeTHLinear
end

@functor HGFBertForSequenceClassification

function HGFBertForSequenceClassification(config::HGFBertConfig)
  bert = HGFBertModel(config)
  classifier = FakeTHLinear(config, config.hidden_size, config.num_labels)
  HGFBertForSequenceClassification(bert, classifier)
end

# multiple choice

struct HGFBertForMultipleChoice <: HGFBertPreTrainedModel
  bert::HGFBertModel
  classifier::FakeTHLinear
end

@functor HGFBertForMultipleChoice

function HGFBertForMultipleChoice(config::HGFBertConfig)
  bert = HGFBertModel(config)
  classifier = FakeTHLinear(config, config.hidden_size, 1)
  HGFBertForMultipleChoice(bert, classifier)
end

# token classify

struct HGFBertForTokenClassification <: HGFBertPreTrainedModel
  bert::HGFBertModel
  classifier::FakeTHLinear
end

@functor HGFBertForTokenClassification

function HGFBertForTokenClassification(config::HGFBertConfig)
  bert = HGFBertModel(config)
  classifier = FakeTHLinear(config, config.hidden_size, config.num_labels)
  HGFBertForTokenClassification(bert, classifier)
end

# qa

struct HGFBertForQuestionAnswering <: HGFBertPreTrainedModel
  bert::HGFBertModel
  qa_outputs::FakeTHLinear
end

@functor HGFBertForQuestionAnswering

function HGFBertForQuestionAnswering(config::HGFBertConfig)
  bert = HGFBertModel(config)
  classifier = FakeTHLinear(config, config.hidden_size, config.num_labels)
  HGFBertForQuestionAnswering(bert, classifier)
end
