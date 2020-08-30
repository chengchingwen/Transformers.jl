# Roberta specific initializers
# same as bert

# HGF Roberta compounts

# embedding

struct HGFRobertaEmbeddings{
  W<:FakeTHEmbedding,
  P<:FakeTHEmbedding,
  T<:FakeTHEmbedding,
  L<:FakeTHLayerNorm
} <: THModule
  LayerNorm::L
  word_embeddings::W
  position_embeddings::P
  token_type_embeddings::T
end

@functor HGFRobertaEmbeddings

function HGFRobertaEmbeddings(config::HGFRobertaConfig)
  layernorm = FakeTHLayerNorm(HGFBertConfig, config, config.hidden_size; eps=config.layer_norm_eps)

  word_emb = FakeTHEmbedding(HGFBertConfig, config, config.vocab_size,
                             config.hidden_size;
                             pad_idx=config.pad_token_id)
  posi_emb = FakeTHEmbedding(HGFBertConfig, config, config.max_position_embeddings,
                             config.hidden_size;
                             pad_idx=config.pad_token_id)
  toke_emb = FakeTHEmbedding(HGFBertConfig, config, config.type_vocab_size,
                             config.hidden_size)

  HGFRobertaEmbeddings(layernorm, word_emb, posi_emb, toke_emb)
end

@inline get_word_emb(be::HGFRobertaEmbeddings, input_ids::AbstractArray{<:Integer}) = be.word_embeddings(input_ids)
@inline get_word_emb(be::HGFRobertaEmbeddings, input_embed::AbstractArray{T}) where T = input_embed

function _create_position_ids_from_input_ids(input_ids, pad_idx)
  mask = input_ids .!= pad_idx
  inc_indices = cumsum(mask; dims=1) .* mask
  return inc_indices .+ pad_idx
end

Flux.@nograd _create_position_ids_from_input_ids

@inline function get_position_emb(be::HGFRobertaEmbeddings, input_ids::AbstractArray{<:Integer}, ::Nothing)
  position_ids = _create_position_ids_from_input_ids(input_ids, be.position_embeddings.pad_idx)
  get_position_emb(be, input_ids, position_ids)
end

@inline function get_position_emb(be::HGFRobertaEmbeddings, inputs_embeds, ::Nothing)
  position_ids = _arange(inputs_embeds, size(inputs_embeds)[end-1]) .+ be.position_embeddings.pad_idx
  get_position_emb(be, inputs_embeds, position_ids)
end

@inline get_position_emb(be::HGFRobertaEmbeddings, inputs_embeds, position_ids) = be.position_embeddings(position_ids)

@inline get_token_type_emb(be::HGFRobertaEmbeddings, inputs_embeds, ::Nothing) = get_token_type_emb(be, inputs_embeds, fill!(similar(inputs_embeds, Int, Base.tail(size(inputs_embeds))), one(Int)))
@inline get_token_type_emb(be::HGFRobertaEmbeddings, inputs_embeds, token_type_ids) = be.token_type_embeddings(token_type_ids)

function (be::HGFRobertaEmbeddings)(input,
                                 position_ids::Union{Nothing, AbstractArray{<:Integer}},
                                 token_type_ids::Union{Nothing, AbstractArray{<:Integer}})
  position_embeds = get_position_emb(be, input, position_ids)
  inputs_embeds = get_word_emb(be, input)
  token_type_embeds = get_token_type_emb(be, inputs_embeds, token_type_ids)
  return be(inputs_embeds, position_embeds, token_type_embeds)
end

function (be::HGFRobertaEmbeddings)(inputs_embeds::AbstractArray{T},
                                 position_embeds::AbstractArray{T},
                                 token_type_embeds::AbstractArray{T}) where T
  embeddings = (inputs_embeds .+ position_embeds) + token_type_embeds
  embeddings = be.LayerNorm(embeddings)
  return embeddings
end

(be::HGFRobertaEmbeddings)(input; position_ids=nothing, token_type_ids=nothing) = be(input, position_ids, token_type_ids)

# roberta model without prediction

abstract type HGFRobertaPreTrainedModel <: HGFBertPreTrainedModel end

struct HGFRobertaModel{
  E<:HGFRobertaEmbeddings,
  T<:HGFBertEncoder,
  P<:HGFBertPooler
} <: HGFRobertaPreTrainedModel
  embeddings::E
  encoder::T
  pooler::P
end

@functor HGFRobertaModel

(bm::HGFRobertaModel)(input; position_ids = nothing, token_type_ids = nothing,
                   attention_mask = nothing,
                   output_attentions = false,
                   output_hidden_states = false
                   ) = bm(input, position_ids, token_type_ids, attention_mask,
                          Val(output_attentions), Val(output_hidden_states))

function (bm::HGFRobertaModel)(
  input, position_ids, token_type_ids,
  attention_mask,
  _output_attentions::Val{output_attentions},
  _output_hidden_states::Val{output_hidden_states}
) where {output_attentions, output_hidden_states}

  embedding_output = bm.embeddings(input, position_ids, token_type_ids)
  attention_mask = create_attention_mask(embedding_output, attention_mask)

  encoder_outputs = bm.encoder(embedding_output, attention_mask,
                               _output_attentions, _output_hidden_states)

  sequence_output = encoder_outputs.last_hidden_state
  pooled_output = bm.pooler(sequence_output)
  return (
    last_hidden_state = sequence_output,
    pooler_output = pooled_output,
    hidden_states = encoder_outputs.hidden_states,
    attentions = encoder_outputs.attentions
  )
end

(bm::HGFRobertaModel)(input, encoder_hidden_states;
                   position_ids = nothing, token_type_ids = nothing,
                   attention_mask = nothing, encoder_attention_mask = nothing,
                   output_attentions = false,
                   output_hidden_states = false
                   ) = bm(input, encoder_hidden_states,
                          position_ids, token_type_ids,
                          attention_mask, encoder_attention_mask,
                          Val(output_attentions), Val(output_hidden_states))

function (bm::HGFRobertaModel)(
  input, encoder_hidden_states, position_ids, token_type_ids,
  attention_mask, encoder_attention_mask,
  _output_attentions::Val{output_attentions},
  _output_hidden_states::Val{output_hidden_states}
) where {output_attentions, output_hidden_states}

  embedding_output = bm.embeddings(input, position_ids, token_type_ids)
  attention_mask = create_causal_attention_mask(embedding_output, attention_mask)
  encoder_attention_mask = create_attention_mask(encoder_hidden_states, encoder_attention_mask)

  encoder_outputs = bm.encoder(embedding_output, attention_mask,
                               encoder_hidden_states, encoder_attention_mask,
                               _output_attentions, _output_hidden_states)

  sequence_output = encoder_outputs.last_hidden_state
  pooled_output = bm.pooler(sequence_output)

  return (
    last_hidden_state = sequence_output,
    pooler_output = pooled_output,
    hidden_states = encoder_outputs.hidden_states,
    attentions = encoder_outputs.attentions
  )
end

function HGFRobertaModel(config::HGFRobertaConfig)
  embeddings = HGFRobertaEmbeddings(config)
  encoder = HGFBertEncoder(HGFBertConfig, config)
  pooler = HGFBertPooler(HGFBertConfig, config)
  HGFRobertaModel(embeddings, encoder, pooler)
end

get_input_embedding(model::HGFRobertaModel) = model.embeddings.word_embeddings.weight

# lm head

struct HGFRobertaLMHead{
  L<:FakeTHLinear,
  N<:FakeTHLayerNorm,
  D<:FakeTHLinear,
  B<:AbstractArray
} <: THModule
  dense::L
  layer_norm::N
  decoder::D
  bias::B
end

@functor HGFRobertaLMHead

function (self::HGFRobertaLMHead)(x)
  x = self.dense(x)
  x = gelu.(x)
  x = self.layer_norm(x)
  x = self.decoder(x) .+ self.bias
  return x
end

function HGFRobertaLMHead(config::HGFRobertaConfig)
  dense = FakeTHLinear(HGFBertConfig, config,
                       config.hidden_size,
                       config.hidden_size)
  layer_norm = FakeTHLayerNorm(HGFBertConfig, config,
                               config.hidden_size;
                               eps=config.layer_norm_eps)
  decoder = FakeTHLinear(HGFBertConfig, config,
                         config.hidden_size,
                         config.vocab_size;
                         bias=false)
  bias = zeros(Float32, config.vocab_size)
  return HGFRobertaLMHead(dense, layer_norm, decoder, bias)
end

# classification head

struct HGFRobertaClassificationHead{
  D<:FakeTHLinear,
  O<:FakeTHLinear} <: THModule
  dense::D
  out_proj::O
end

@functor HGFRobertaClassificationHead

function (self::HGFRobertaClassificationHead)(features)
  x = features[:, 1, :]
  x = self.dense(x)
  x = tanh.(x)
  x = self.out_proj
end

function HGFRobertaClassificationHead(config::HGFRobertaConfig)
  dense = FakeTHLinear(HGFBertConfig, config,
                       config.hidden_size,
                       config.hidden_size)
  out_proj = FakeTHLinear(HGFBertConfig, config,
                          config.hidden_size,
                          config.num_labels)
  return HGFRobertaClassificationHead(dense, out_proj)
end

# roberta models for different task

# causal lm

struct HGFRobertaForCausalLM{
  B<:HGFRobertaModel,
  L<:HGFRobertaLMHead
} <: HGFRobertaPreTrainedModel
  roberta::B
  lm_head::L
end

@functor HGFRobertaForCausalLM

(self::HGFRobertaForCausalLM)(input;
                              position_ids = nothing, token_type_ids = nothing,
                              attention_mask = nothing,
                              output_attentions = false,
                              output_hidden_states = false
                              ) = self(input, position_ids, token_type_ids,
                                       attention_mask,
                                       Val(output_attentions), Val(output_hidden_states))

function (self::HGFRobertaForCausalLM)(input, position_ids, token_type_ids,
                                       attention_mask,
                                       _output_attentions::Val{output_attentions},
                                       _output_hidden_states::Val{output_hidden_states}
                                       ) where {output_attentions, output_hidden_states}
  outputs = self.roberta(input, position_ids, token_type_ids,
                         attention_mask, _output_attentions, _output_hidden_states)
  sequence_output = outputs[1]
  prediction_scores = self.lm_head(sequence_output)
  lm_loss = nothing

  return (
    loss = lm_loss,
    logits = prediction_scores,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

(self::HGFRobertaForCausalLM)(input, labels;
                              position_ids = nothing, token_type_ids = nothing,
                              attention_mask = nothing,
                              output_attentions = false,
                              output_hidden_states = false
                              ) = self(input, labels,
                                       position_ids, token_type_ids,
                                       attention_mask,
                                       Val(output_attentions), Val(output_hidden_states))

function (self::HGFRobertaForCausalLM)(input, labels,
                                    position_ids, token_type_ids,
                                       attention_mask,
                                       _output_attentions::Val{output_attentions},
                                       _output_hidden_states::Val{output_hidden_states}
                                       ) where {output_attentions, output_hidden_states}
  outputs = self(input, position_ids, token_type_ids, attention_mask,
                 _output_attentions, _output_hidden_states)

  prediction_scores = outputs.prediction_logits

  shifted_prediction_scores = prediction_scores[:, 1:end-1, :]
  shifted_labels = labels[:, 2:end]

  lm_loss = Flux.logitcrossentropy(shifted_prediction_scores, shifted_labels)

  return (
    loss = lm_loss,
    logits = prediction_scores,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

function HGFRobertaForCausalLM(config::HGFRobertaConfig)
  roberta = HGFRobertaModel(config)
  lm_head = HGFRobertaLMHead(config)
  return HGFRobertaForCausalLM(roberta, lm_head)
end

# masked lm

struct HGFRobertaForMaskedLM{
  B<:HGFRobertaModel,
  L<:HGFRobertaLMHead
} <: HGFRobertaPreTrainedModel
  roberta::B
  lm_head::L
end

@functor HGFRobertaForMaskedLM

(self::HGFRobertaForMaskedLM)(input;
                              position_ids = nothing, token_type_ids = nothing,
                              attention_mask = nothing,
                              output_attentions = false,
                              output_hidden_states = false
                              ) = self(input, position_ids, token_type_ids,
                                       attention_mask,
                                       Val(output_attentions), Val(output_hidden_states))

function (self::HGFRobertaForMaskedLM)(input, position_ids, token_type_ids,
                                       attention_mask,
                                       _output_attentions::Val{output_attentions},
                                       _output_hidden_states::Val{output_hidden_states}
                                       ) where {output_attentions, output_hidden_states}
  outputs = self.roberta(input, position_ids, token_type_ids,
                         attention_mask, _output_attentions, _output_hidden_states)
  sequence_output = outputs[1]
  prediction_scores = self.lm_head(sequence_output)
  lm_loss = nothing

  return (
    loss = lm_loss,
    logits = prediction_scores,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

(self::HGFRobertaForMaskedLM)(input, labels;
                              position_ids = nothing, token_type_ids = nothing,
                              attention_mask = nothing,
                              output_attentions = false,
                              output_hidden_states = false
                              ) = self(input, labels,
                                       position_ids, token_type_ids,
                                       attention_mask,
                                       Val(output_attentions), Val(output_hidden_states))

function (self::HGFRobertaForMaskedLM)(input, labels,
                                    position_ids, token_type_ids,
                                       attention_mask,
                                       _output_attentions::Val{output_attentions},
                                       _output_hidden_states::Val{output_hidden_states}
                                       ) where {output_attentions, output_hidden_states}
  outputs = self(input, position_ids, token_type_ids, attention_mask,
                 _output_attentions, _output_hidden_states)

  prediction_scores = outputs.prediction_logits

  lm_loss = Flux.logitcrossentropy(prediction_scores, labels)

  return (
    loss = lm_loss,
    logits = prediction_scores,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

function HGFRobertaForMaskedLM(config::HGFRobertaConfig)
  roberta = HGFRobertaModel(config)
  lm_head = HGFRobertaLMHead(config)
  return HGFRobertaForMaskedLM(roberta, lm_head)
end

# sequence classification

struct HGFRobertaForSequenceClassification{
  B<:HGFRobertaModel,
  C<:HGFRobertaClassificationHead
} <: HGFRobertaPreTrainedModel
  roberta::B
  classifier::C
end

@functor HGFRobertaForSequenceClassification

(self::HGFRobertaForSequenceClassification)(input;
                              position_ids = nothing, token_type_ids = nothing,
                              attention_mask = nothing,
                              output_attentions = false,
                              output_hidden_states = false
                              ) = self(input, position_ids, token_type_ids,
                                       attention_mask,
                                       Val(output_attentions), Val(output_hidden_states))

function (self::HGFRobertaForSequenceClassification)(input, position_ids, token_type_ids,
                                       attention_mask,
                                       _output_attentions::Val{output_attentions},
                                       _output_hidden_states::Val{output_hidden_states}
                                       ) where {output_attentions, output_hidden_states}
  outputs = self.roberta(input, position_ids, token_type_ids,
                         attention_mask, _output_attentions, _output_hidden_states)
  sequence_output = outputs[1]
  logits = self.classifier(sequence_output)
  loss = nothing

  return (
    loss = loss,
    logits = logits,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

(self::HGFRobertaForSequenceClassification)(input, labels;
                              position_ids = nothing, token_type_ids = nothing,
                              attention_mask = nothing,
                              output_attentions = false,
                              output_hidden_states = false
                              ) = self(input, labels,
                                       position_ids, token_type_ids,
                                       attention_mask,
                                       Val(output_attentions), Val(output_hidden_states))

function (self::HGFRobertaForSequenceClassification)(input, labels,
                                    position_ids, token_type_ids,
                                       attention_mask,
                                       _output_attentions::Val{output_attentions},
                                       _output_hidden_states::Val{output_hidden_states}
                                       ) where {output_attentions, output_hidden_states}
  outputs = self(input, position_ids, token_type_ids, attention_mask,
                 _output_attentions, _output_hidden_states)

  logits = outputs.logits
  if size(logits, 1) == 1
    loss = Flux.mse(logits, labels)
  else
    loss = Flux.logitcrossentropy(logits, labels)
  end

  return (
    loss = loss,
    logits = logits,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

function HGFRobertaForSequenceClassification(config::HGFRobertaConfig)
  roberta = HGFRobertaModel(config)
  classifier = HGFRobertaClassificationHead(config)
  return HGFRobertaForSequenceClassification(roberta, classifier)
end

# multiple chhoice

struct HGFRobertaForMultipleChoice{
  B<:HGFRobertaModel,
  C<:FakeTHLinear
} <: HGFRobertaPreTrainedModel
  roberta::B
  classifier::C
end

@functor HGFRobertaForMultipleChoice

(self::HGFRobertaForMultipleChoice)(input;
                              position_ids = nothing, token_type_ids = nothing,
                              attention_mask = nothing,
                              output_attentions = false,
                              output_hidden_states = false
                              ) = self(input, position_ids, token_type_ids,
                                       attention_mask,
                                       Val(output_attentions), Val(output_hidden_states))

function (self::HGFRobertaForMultipleChoice)(input, position_ids, token_type_ids,
                                    attention_mask,
                                    _output_attentions::Val{output_attentions},
                                    _output_hidden_states::Val{output_hidden_states}
                                          ) where {output_attentions, output_hidden_states}

  num_choices = size(input, ndims(input)-1)
  flat_choice(x) = reshape(x, size(x)[1:end-2]..., :)
  flat_choice(::Nothing) = nothing
  flat_input = flat_choice(input)
  flat_position_ids = flat_choice(position_ids)
  flat_token_type_ids = flat_choice(token_type_ids)
  flat_attention_mask = flat_choice(attention_mask)

  outputs = self.roberta(flat_input, flat_position_ids, flat_token_type_ids,
                      flat_attention_mask, _output_attentions, _output_hidden_states)

  pooled_output = outputs[2]
  logits = self.classifier(pooled_output)
  reshaped_logits = reshape(logits, num_choices, :)
  loss = nothing

  return (
    loss = loss,
    logits = reshaped_logits,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

(self::HGFRobertaForMultipleChoice)(input, labels;
                           position_ids = nothing, token_type_ids = nothing,
                           attention_mask = nothing,
                           output_attentions = false,
                           output_hidden_states = false
                           ) = self(input, labels,
                                    position_ids, token_type_ids,
                                    attention_mask,
                                    Val(output_attentions), Val(output_hidden_states))

function (self::HGFRobertaForMultipleChoice)(input, labels,
                                          position_ids, token_type_ids,
                                          attention_mask,
                                          _output_attentions::Val{output_attentions},
                                          _output_hidden_states::Val{output_hidden_states}
                                          ) where {output_attentions, output_hidden_states}
  outputs = self(input, position_ids, token_type_ids, attention_mask,
                 _output_attentions, _output_hidden_states)

  logits = outputs.logits
  loss = Flux.logitcrossentropy(logits, labels)

  return (
    loss = loss,
    logits = logits,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

function HGFRobertaForMultipleChoice(config::HGFRobertaConfig)
  roberta = HGFRobertaModel(config)
  classifier = FakeTHLinear(HGFBertConfig, config,
                            config.hidden_size, 1)
  return HGFRobertaForMultipleChoice(roberta, classifier)
end

# token classification

struct HGFRobertaForTokenClassification{
  B<:HGFRobertaModel,
  C<:FakeTHLinear
} <: HGFRobertaPreTrainedModel
  roberta::B
  classifier::C
end

@functor HGFRobertaForTokenClassification

(self::HGFRobertaForTokenClassification)(input;
                              position_ids = nothing, token_type_ids = nothing,
                              attention_mask = nothing,
                              output_attentions = false,
                              output_hidden_states = false
                              ) = self(input, position_ids, token_type_ids,
                                       attention_mask,
                                       Val(output_attentions), Val(output_hidden_states))

function (self::HGFRobertaForTokenClassification)(input, position_ids, token_type_ids,
                                    attention_mask,
                                    _output_attentions::Val{output_attentions},
                                    _output_hidden_states::Val{output_hidden_states}
                                    ) where {output_attentions, output_hidden_states}
  outputs = self.roberta(input, position_ids, token_type_ids,
                      attention_mask, _output_attentions, _output_hidden_states)
  sequence_output = outputs[1]
  logits = self.classifier(sequence_output)
  loss = nothing

  return (
    loss = loss,
    logits = logits,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

(self::HGFRobertaForTokenClassification)(input, labels;
                           position_ids = nothing, token_type_ids = nothing,
                           attention_mask = nothing,
                           output_attentions = false,
                           output_hidden_states = false
                           ) = self(input, labels,
                                    position_ids, token_type_ids,
                                    attention_mask,
                                    Val(output_attentions), Val(output_hidden_states))

function (self::HGFRobertaForTokenClassification)(input, labels,
                                    position_ids, token_type_ids,
                                       attention_mask,
                                       _output_attentions::Val{output_attentions},
                                       _output_hidden_states::Val{output_hidden_states}
                                       ) where {output_attentions, output_hidden_states}
  outputs = self(input, position_ids, token_type_ids, attention_mask,
                _output_attentions, _output_hidden_states)

  logits = outputs.logits
  loss = Flux.logitcrossentropy(logits, labels)

  return (
    loss = loss,
    logits = logits,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

function HGFRobertaForTokenClassification(config::HGFRobertaConfig)
  roberta = HGFRobertaModel(config)
  classifier = FakeTHLinear(HGFBertConfig, config,
                            config.hidden_size,
                            config.num_labels)
  return HGFRobertaForTokenClassification(roberta, classifier)
end

# question answering

struct HGFRobertaForQuestionAnswering{
  B<:HGFRobertaModel,
  C<:FakeTHLinear
} <: HGFRobertaPreTrainedModel
  roberta::B
  qa_outputs::C
end

@functor HGFRobertaForQuestionAnswering

(self::HGFRobertaForQuestionAnswering)(input;
                              position_ids = nothing, token_type_ids = nothing,
                              attention_mask = nothing,
                              output_attentions = false,
                              output_hidden_states = false
                              ) = self(input, position_ids, token_type_ids,
                                       attention_mask,
                                       Val(output_attentions), Val(output_hidden_states))

function (self::HGFRobertaForQuestionAnswering)(input, position_ids, token_type_ids,
                                    attention_mask,
                                    _output_attentions::Val{output_attentions},
                                    _output_hidden_states::Val{output_hidden_states}
                                    ) where {output_attentions, output_hidden_states}
  outputs = self.roberta(input, position_ids, token_type_ids,
                      attention_mask, _output_attentions, _output_hidden_states)
  sequence_output = outputs[1]
  logits = self.qa_outputs(sequence_output)
  start_logits = @view(logits[1, :])
  end_logits = @view(logits[2, :])
  total_loss = nothing

  return (
    loss = total_loss,
    start_logits = start_logits,
    end_logits = end_logits,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

(self::HGFRobertaForQuestionAnswering)(input, start_positions, end_positions;
                           position_ids = nothing, token_type_ids = nothing,
                           attention_mask = nothing,
                           output_attentions = false,
                           output_hidden_states = false
                           ) = self(input, start_positions, end_positions,
                                    position_ids, token_type_ids,
                                    attention_mask,
                                    Val(output_attentions), Val(output_hidden_states))

function (self::HGFRobertaForQuestionAnswering)(input, start_positions, end_positions,
                                    position_ids, token_type_ids,
                                       attention_mask,
                                       _output_attentions::Val{output_attentions},
                                       _output_hidden_states::Val{output_hidden_states}
                                       ) where {output_attentions, output_hidden_states}
  outputs = self(input, position_ids, token_type_ids, attention_mask,
                _output_attentions, _output_hidden_states)

  start_logits = outputs.start_logits
  end_logits = outputs.end_logits
  start_loss = Flux.logitcrossentropy(start_logits, start_positions)
  end_loss = Flux.logitcrossentropy(end_logits, end_positions)
  total_loss = (start_loss + end_loss) / 2

  return (
    loss = total_loss,
    start_logits = start_logits,
    end_logits = end_logits,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

function HGFRobertaForQuestionAnswering(config::HGFRobertaConfig)
  roberta = HGFRobertaModel(config)
  qa_outputs = FakeTHLinear(HGFBertConfig, config,
                            config.hidden_size,
                            config.num_labels)
  return HGFRobertaForQuestionAnswering(roberta, qa_outputs)
end

# load model utils

basemodelkey(::HGFRobertaPreTrainedModel) = :roberta
basemodel(m::HGFRobertaPreTrainedModel) = getproperty(m, basemodelkey(m))
basemodel(m::HGFRobertaModel) = m

isbasemodel(m::HGFRobertaModel) = true
isbasemodel(m::HGFRobertaPreTrainedModel) = false

get_model_type(::Val{:roberta}, ::Val{:model}) = HGFRobertaModel
get_model_type(::Val{:roberta}, ::Val{:forcausallm}) = HGFRobertaForCausalLM
get_model_type(::Val{:roberta}, ::Val{:formaskedlm}) = HGFRobertaForMaskedLM
get_model_type(::Val{:roberta}, ::Val{:forsequenceclassification}) = HGFRobertaForSequenceClassification
get_model_type(::Val{:roberta}, ::Val{:formultiplechoice}) = HGFRobertaForMultipleChoice
get_model_type(::Val{:roberta}, ::Val{:fortokenclassification}) = HGFRobertaForTokenClassification
get_model_type(::Val{:roberta}, ::Val{:forquestionanswering}) = HGFRobertaForQuestionAnswering

