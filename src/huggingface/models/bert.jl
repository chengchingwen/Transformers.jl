using ..Transformers: batchedmul, batched_triu!

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
  weight = randn(Float32, dims, num) .* config.initializer_range
  if !isnothing(pad_idx)
    real_pad_idx = pad_idx+1
  else
    real_pad_idx = 0
  end
  FakeTHEmbedding(real_pad_idx, weight)
end

function FakeTHLayerNorm(config::HGFBertConfig, dims; eps::Float32=1e-05)
  weight = ones(Float32, dims)
  bias = zeros(Float32, dims)
  FakeTHLayerNorm(eps, weight, bias)
end

# HGF Bert compounts

# embedding

struct HGFBertEmbeddings{
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

_arange(x, len) = cumsum(fill!(similar(x, Int, len), one(Int)))

Flux.@nograd _arange

@inline get_word_emb(be::HGFBertEmbeddings, input_ids::AbstractArray{<:Integer}) = be.word_embeddings(input_ids)
@inline get_word_emb(be::HGFBertEmbeddings, input_embed::AbstractArray{T}) where T = input_embed

@inline get_position_emb(be::HGFBertEmbeddings, inputs_embeds, ::Nothing) = get_position_emb(be, inputs_embeds, _arange(inputs_embeds, size(inputs_embeds)[end-1]))
@inline get_position_emb(be::HGFBertEmbeddings, inputs_embeds, position_ids) = be.position_embeddings(position_ids)

@inline get_token_type_emb(be::HGFBertEmbeddings, inputs_embeds, ::Nothing) = get_token_type_emb(be, inputs_embeds, fill!(similar(inputs_embeds, Int, Base.tail(size(inputs_embeds))), one(Int)))
@inline get_token_type_emb(be::HGFBertEmbeddings, inputs_embeds, token_type_ids) = be.token_type_embeddings(token_type_ids)

function (be::HGFBertEmbeddings)(input,
                                 position_ids::Union{Nothing, AbstractArray{<:Integer}},
                                 token_type_ids::Union{Nothing, AbstractArray{<:Integer}})

  inputs_embeds = get_word_emb(be, input)
  position_embeds = get_position_emb(be, inputs_embeds, position_ids)
  token_type_embeds = get_token_type_emb(be, inputs_embeds, token_type_ids)
  return be(inputs_embeds, position_embeds, token_type_embeds)
end

function (be::HGFBertEmbeddings)(inputs_embeds::AbstractArray{T},
                                 position_embeds::AbstractArray{T},
                                 token_type_embeds::AbstractArray{T}) where T
  embeddings = (inputs_embeds .+ position_embeds) + token_type_embeds
  embeddings = be.LayerNorm(embeddings)
  return embeddings
end

(be::HGFBertEmbeddings)(input; position_ids=nothing, token_type_ids=nothing) = be(input, position_ids, token_type_ids)

# self attention part

struct HGFBertSelfAttention{
  Q<:FakeTHLinear,
  K<:FakeTHLinear,
  V<:FakeTHLinear
} <: THModule
  num_attention_heads::Int
  query::Q
  key::K
  value::V
end

@functor HGFBertSelfAttention

function _split_tranpose_for_scores(x, num_head)
  head_size = div(size(x, 1), num_head)
  split_size = (head_size, num_head, size(x, 2), size(x, 3))
  permute_order = (1, 3, 2, 4)

  return reshape(x, split_size) |> Base.Fix2(permutedims, permute_order)
end

function _compute_attention_scores(query_layer, key_layer, attention_mask::Union{Nothing, <:AbstractArray})
  attentions_scores = batchedmul(key_layer, query_layer; transA = true)
  attentions_scores = attentions_scores ./ convert(eltype(attentions_scores), sqrt(size(key_layer, 1)))

  !isnothing(attention_mask) &&
    (attentions_scores = attentions_scores .+ attention_mask)

  return attentions_scores
end

function _merge_transpose_for_output(x, num_head)
  permute_order = (1, 3, 2, 4)
  final_size = (:, size(x, 2), size(x, 4))

  return permutedims(x, permute_order) |>
    Base.Fix2(reshape, final_size)
end

function (sa::HGFBertSelfAttention)(hidden_states, attention_mask, output_attentions::Val)
  mixed_query_layer = sa.query(hidden_states)
  mixed_key_layer = sa.key(hidden_states)
  mixed_value_layer = sa.value(hidden_states)

  return sa(mixed_query_layer, mixed_key_layer, mixed_value_layer, attention_mask, output_attentions)
end

function (sa::HGFBertSelfAttention)(hidden_states, encoder_hidden_states, attention_mask, output_attentions::Val)
  mixed_query_layer = sa.query(hidden_states)
  mixed_key_layer = sa.key(encoder_hidden_states)
  mixed_value_layer = sa.value(encoder_hidden_states)

  return sa(mixed_query_layer, mixed_key_layer, mixed_value_layer, attention_mask, output_attentions)
end

function (sa::HGFBertSelfAttention)(mixed_query_layer, mixed_key_layer, mixed_value_layer,
                                    attention_mask,
                                    ::Val{output_attentions}) where output_attentions

  query_layer = _split_tranpose_for_scores(mixed_query_layer, sa.num_attention_heads)
  key_layer = _split_tranpose_for_scores(mixed_key_layer, sa.num_attention_heads)
  value_layer = _split_tranpose_for_scores(mixed_value_layer, sa.num_attention_heads)

  attentions_scores = _compute_attention_scores(query_layer, key_layer, attention_mask)
  attentions_probs = softmax(attentions_scores; dims=1)

  mixed_context_layer = batchedmul(value_layer, attentions_probs)
  context_layer = _merge_transpose_for_output(mixed_context_layer, sa.num_attention_heads)

  if output_attentions
    outputs = context_layer, attentions_probs
    return outputs
  else
    return context_layer
  end
end

function HGFBertSelfAttention(config::HGFBertConfig)
  attention_head_size = config.hidden_size ÷ config.num_attention_heads
  all_head_size = config.num_attention_heads * attention_head_size

  query = FakeTHLinear(config, config.hidden_size, all_head_size)
  key   = FakeTHLinear(config, config.hidden_size, all_head_size)
  value = FakeTHLinear(config, config.hidden_size, all_head_size)

  HGFBertSelfAttention(config.num_attention_heads, query, key, value)
end

# self attention output part

struct HGFBertSelfOutput{
  L<:FakeTHLayerNorm,
  D<:FakeTHLinear
} <: THModule
  LayerNorm::L
  dense::D
end

@functor HGFBertSelfOutput

function (so::HGFBertSelfOutput)(hidden_states, input_tensor)
  hidden_states = so.dense(hidden_states)
  hidden_states = so.LayerNorm(hidden_states + input_tensor)
  return hidden_states
end

function HGFBertSelfOutput(config::HGFBertConfig)
  layernorm = FakeTHLayerNorm(config, config.hidden_size; eps=config.layer_norm_eps)
  dense = FakeTHLinear(config, config.hidden_size, config.hidden_size)
  HGFBertSelfOutput(layernorm, dense)
end

# self attention

struct HGFBertAttention{
  S<:HGFBertSelfAttention,
  O<:HGFBertSelfOutput
} <: THModule
  self::S
  output::O
end

@functor HGFBertAttention

function (a::HGFBertAttention)(hidden_states,
                               attention_mask::Union{Nothing, <:AbstractArray},
                               _output_attentions::Val{output_attentions}) where output_attentions

  self_output = a.self(hidden_states, attention_mask, _output_attentions)

  if output_attentions
    output, attention_prob = self_output
    attention_output = a.output(output, hidden_states)
    return attention_output, attention_prob
  else
    attention_output = a.output(self_output, hidden_states)
    return attention_output
  end
end

function HGFBertAttention(config::HGFBertConfig)
  self = HGFBertSelfAttention(config)
  output = HGFBertSelfOutput(config)
  HGFBertAttention(self, output)
end

# positionwise first dense

struct HGFBertIntermediate{F, D<:FakeTHLinear} <: THModule
  intermediate_act::F
  dense::D
end

Functors.functor(::Type{<:HGFBertIntermediate}, intermediate) = (dense = intermediate.dense,), y->HGFBertIntermediate(intermediate.intermediate_act, y...)

(i::HGFBertIntermediate)(hidden_states) = i.intermediate_act.(i.dense(hidden_states))

function HGFBertIntermediate(config::HGFBertConfig)
  global ACT2FN
  act = ACT2FN[Symbol(config.hidden_act)]
  dense = FakeTHLinear(config, config.hidden_size, config.intermediate_size)
  HGFBertIntermediate(act, dense)
end

# positionwise second dense

struct HGFBertOutput{
  D<:FakeTHLinear,
  L<:FakeTHLayerNorm
} <: THModule
  dense::D
  LayerNorm::L
end

@functor HGFBertOutput

function (o::HGFBertOutput)(hidden_states, input_tensor)
  hidden_states = o.dense(hidden_states)
  hidden_states = o.LayerNorm(hidden_states + input_tensor)
  return hidden_states
end

function HGFBertOutput(config::HGFBertConfig)
  dense = FakeTHLinear(config, config.intermediate_size, config.hidden_size)
  layernorm = FakeTHLayerNorm(config, config.hidden_size; eps=config.layer_norm_eps)
  HGFBertOutput(dense, layernorm)
end

# transformer layer

struct HGFBertLayer{DEC<:Union{Nothing, HGFBertAttention},
                    A<:HGFBertAttention,
                    I<:HGFBertIntermediate,
                    O<:HGFBertOutput
} <: THModule
  attention::A
  crossattention::DEC
  intermediate::I
  output::O
end

HGFBertLayer(a, i, o) = HGFBertLayer(a, nothing, i, o)

_is_decode(::HGFBertLayer{Nothing}) = false
_is_decode(::HGFBertLayer) = true

Functors.functor(::Type{<:HGFBertLayer}, layer) = (!_is_decode(layer) ?
    (attention = layer.attention, intermediate = layer.intermediate, output = layer.output) :
    (attention = layer.attention, crossattention = layer.crossattention, intermediate = layer.intermediate, output = layer.output)),
    y ->HGFBertLayer(y...)

function (l::HGFBertLayer{Nothing})(hidden_states, attention_mask::Union{Nothing, <:AbstractArray},
                                    _output_attentions::Val{output_attentions}) where output_attentions
  if output_attentions
    attention_output, attention_prob = l.attention(hidden_states, attention_mask, _output_attentions)
  else
    attention_output = l.attention(hidden_states, attention_mask, _output_attentions)
  end

  intermediate_output = l.intermediate(attention_output)
  layer_output = l.output(intermediate_output, attention_output)

  if output_attentions
    return layer_output, attention_prob
  else
    return layer_output
  end
end

function (l::HGFBertLayer)(hidden_states, attention_mask,
                           encoder_hidden_states, encoder_attention_mask,
                           _output_attentions::Val{output_attentions}) where output_attentions
  if output_attentions
    attention_output, attention_prob = l.attention(hidden_states, attention_mask, _output_attentions)
    attention_output, cross_attention_prob = l.crossattention(attention_output, encoder_attention_mask, _output_attentions)
  else
    attention_output = l.attention(hidden_states, attention_mask, _output_attentions)
    attention_output = l.crossattention(attention_output, encoder_hidden_states, encoder_attention_mask, _output_attentions)
  end

  intermediate_output = l.intermediate(attention_output)
  layer_output = l.output(intermediate_output, attention_output)

  if output_attentions
    return layer_output, attention_prob, cross_attention_prob
  else
    return layer_output
  end
end

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

struct HGFBertEncoder{N, L<:FakeTHModuleList{N}} <: THModule
  layer::L
end

@functor HGFBertEncoder

(e::HGFBertEncoder)(hidden_states;
                    attention_mask = nothing,
                    output_attentions = false,
                    output_hidden_states = false
                    ) = e(hidden_states, attention_mask, Val(output_attentions), Val(output_hidden_states))

@generated function (e::HGFBertEncoder{N})(hidden_states, attention_mask,
                             _output_attentions::Val{output_attentions},
                             _output_hidden_states::Val{output_hidden_states}
                             ) where {N, output_attentions, output_hidden_states}
  if output_attentions
    all_attentions = Expr(:tuple)
  end

  if output_hidden_states
    all_hidden_states = Expr(:tuple, :hidden_states)
  end

  body = Expr[]

  for i = 1:N
    previous = i == 1 ? :hidden_states : Symbol(:hidden_states, i-1)
    current = Symbol(:hidden_states, i)
    if output_attentions
      current_attention = Symbol(:attention_, i)
      current_output = :($current, $current_attention)
    else
      current_output = current
    end

    expr = :($current_output = e.layer[$i]($previous, attention_mask, _output_attentions))
    push!(body, expr)

    if output_attentions
      push!(all_attentions.args, current_attention)
    end

    if output_hidden_states
      push!(all_hidden_states.args, current)
    end
  end

  current = Symbol(:hidden_states, N)

  if output_attentions
    push!(body, :(all_attentions = $all_attentions))
  else
    push!(body, :(all_attentions = nothing))
  end

  if output_hidden_states
    push!(body, :(all_hidden_states = $all_hidden_states))
  else
    push!(body, :(all_hidden_states = nothing))
  end

  return quote
    $(body...)

    return (
      last_hidden_state = $current,
      hidden_states = all_hidden_states,
      attentions = all_attentions
    )
  end
end

(e::HGFBertEncoder)(hidden_states, encoder_hidden_states;
                    attention_mask = nothing,
                    encoder_mask = nothing,
                    output_attentions = false,
                    output_hidden_states = false
                    ) = e(hidden_states, attention_mask,
                          encoder_hidden_states, encoder_mask,
                          Val(output_attentions), Val(output_hidden_states))


@generated function (e::HGFBertEncoder{N})(hidden_states, attention_mask,
                                           encoder_hidden_states, encoder_mask,
                                           _output_attentions::Val{output_attentions},
                                           _output_hidden_states::Val{output_hidden_states}
                                           ) where {N, output_attentions, output_hidden_states}
  if output_attentions
    all_attentions = Expr(:tuple)
  end

  if output_hidden_states
    all_hidden_states = Expr(:tuple, :hidden_states)
  end

  body = Expr[]

  for i = 1:N
    previous = i == 1 ? :hidden_states : Symbol(:hidden_states, i-1)
    current = Symbol(:hidden_states, i)
    if output_attentions
      current_attention = Symbol(:attention_, i)
      current_output = :($current, $current_attention)
    else
      current_output = current
    end

    expr = :($current_output = e.layer[$i]($previous, attention_mask,
                                           encoder_hidden_states, encoder_mask,
                                           _output_attentions))
    push!(body, expr)

    if output_attentions
      push!(all_attentions.args, current_attention)
    end

    if output_hidden_states
      push!(all_hidden_states.args, current)
    end
  end

  current = Symbol(:hidden_states, N)

  if output_attentions
    push!(body, :(all_attentions = $all_attentions))
  else
    push!(body, :(all_attentions = nothing))
  end

  if output_hidden_states
    push!(body, :(all_hidden_states = $all_hidden_states))
  else
    push!(body, :(all_hidden_states = nothing))
  end

  return quote
    $(body...)

    return (
      last_hidden_state = $current,
      hidden_states = all_hidden_states,
      attentions = all_attentions
    )
  end
end

function HGFBertEncoder(config::HGFBertConfig)
  layer = FakeTHModuleList(
    [HGFBertLayer(config) for _ in 1:config.num_hidden_layers]
  )
  HGFBertEncoder(layer)
end

# classify token projection

struct HGFBertPooler{D<:FakeTHLinear} <: THModule
  dense::D
end

@functor HGFBertPooler

function (p::HGFBertPooler)(hidden_states)
  first_token_tensor = hidden_states[:, 1, :]
  pooled_output = tanh.(p.dense(first_token_tensor))
  return pooled_output
end

function HGFBertPooler(config::HGFBertConfig)
  dense = FakeTHLinear(config, config.hidden_size, config.hidden_size)
  HGFBertPooler(dense)
end

# label prediction layer

struct HGFBertPredictionHeadTransform{
  F,
  D<:FakeTHLinear,
  L<:FakeTHLayerNorm
} <: THModule
  transform_act_fn::F
  dense::D
  LayerNorm::L
end

@functor HGFBertPredictionHeadTransform

function (pht::HGFBertPredictionHeadTransform)(hidden_states)
  hidden_states = pht.dense(hidden_states)
  hidden_states = pht.transform_act_fn.(hidden_states)
  hidden_states = pht.LayerNorm(hidden_states)
  return hidden_states
end

function HGFBertPredictionHeadTransform(config::HGFBertConfig)
  global ACT2FN
  dense = FakeTHLinear(config, config.hidden_size, config.hidden_size)
  act = ACT2FN[Symbol(config.hidden_act)]
  layernorm = FakeTHLayerNorm(config, config.hidden_size; eps=config.layer_norm_eps)

  return HGFBertPredictionHeadTransform(act, dense, layernorm)
end

# language model prediction layer

struct HGFBertLMPredictionHead{
  B<:AbstractArray,
  T<:HGFBertPredictionHeadTransform,
  D<:FakeTHLinear
} <: THModule
  transform::T
  decoder::D
  bias::B
end

@functor HGFBertLMPredictionHead

function (ph::HGFBertLMPredictionHead)(hidden_states)
  hidden_states = ph.transform(hidden_states)
  hidden_states = ph.decoder(hidden_states) .+ ph.bias
  return hidden_states
end

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

struct HGFBertOnlyMLMHead{P<:HGFBertLMPredictionHead} <: THModule
  predictions::P
end

@functor HGFBertOnlyMLMHead

(h::HGFBertOnlyMLMHead)(sequence_output) = h.predictions(sequence_output)

function HGFBertOnlyMLMHead(config::HGFBertConfig; input_embedding=nothing)
  predictions = HGFBertLMPredictionHead(config; input_embedding=input_embedding)
  return HGFBertOnlyMLMHead(predictions)
end

# next sentence prediction layer

struct HGFBertOnlyNSPHead{S<:FakeTHLinear} <: THModule
  seq_relationship::S
end

@functor HGFBertOnlyNSPHead

(h::HGFBertOnlyNSPHead)(pooled_output) = h.seq_relationship(pooled_output)

function HGFBertOnlyNSPHead(config::HGFBertConfig)
  seq_relationship = FakeTHLinear(config, config.hidden_size, 2)
  return HGFBertOnlyNSPHead(seq_relationship)
end

# pretrain prediction layers

struct HGFBertPreTrainingHeads{
  P<:HGFBertLMPredictionHead,
  S<:FakeTHLinear
} <: THModule
  predictions::P
  seq_relationship::S
end

@functor HGFBertPreTrainingHeads

(pth::HGFBertPreTrainingHeads)(sequence_output, pooled_output) = pth.predictions(sequence_output), pth.seq_relationship(pooled_output)

function HGFBertPreTrainingHeads(config::HGFBertConfig; input_embedding=nothing)
  predictions = HGFBertLMPredictionHead(config; input_embedding=input_embedding)
  seq_relationship = FakeTHLinear(config, config.hidden_size, 2)
  return HGFBertPreTrainingHeads(predictions, seq_relationship)
end

# bert model without prediction
abstract type HGFBertPreTrainedModel <: HGFPreTrainedModel end

struct HGFBertModel{
  E<:HGFBertEmbeddings,
  T<:HGFBertEncoder,
  P<:HGFBertPooler
} <: HGFBertPreTrainedModel
  embeddings::E
  encoder::T
  pooler::P
end

@functor HGFBertModel

function maybe_prepare_mask(embedding_output, ::Nothing)
  mask_size = size(embedding_output) |> Base.tail
  fill!(similar(embedding_output, mask_size), 1) |> maybe_prepare_mask
end
maybe_prepare_mask(embedding_output, attention_mask) = maybe_prepare_mask(attention_mask)
maybe_prepare_mask(embedding_output, attention_mask::AbstractArray{T, 4}) where T = attention_mask

function maybe_prepare_mask(attention_mask::AbstractArray{T, 3}) where T
  seq_len1, seq_len2, batch_size = size(attention_mask)
  attention_mask = reshape(attention_mask, seq_len1, seq_len2, 1, batch_size)
  return attention_mask
end

function maybe_prepare_mask(attention_mask::AbstractMatrix)
  seq_len, batch_size = size(attention_mask)
  attention_mask = reshape(attention_mask, seq_len, 1, 1, batch_size)
  return attention_mask
end

function create_attention_mask(attention_mask::AbstractArray{T, 4}) where T
  return (one(T) .- attention_mask) .* convert(T, -10000)
end

function create_attention_mask(embedding_output, attention_mask)
  attention_mask = maybe_prepare_mask(embedding_output, attention_mask)
  create_attention_mask(attention_mask)
end

maybe_prepare_causal_mask(embedding_output, attention_mask::AbstractArray{T, 4}) where T = attention_mask
maybe_prepare_causal_mask(embedding_output, attention_mask::AbstractArray{T, 3}) where T = maybe_prepare_mask(embedding_output, attention_mask)
function maybe_prepare_causal_mask(embedding_output, attention_mask::Union{Nothing, AbstractMatrix})
  regular_mask = maybe_prepare_mask(embedding_output, attention_mask) # seq_len, 1, 1, b
  attention_mask = permutedims(regular_mask, (2,1,3,4)) .* regular_mask # seq_len, seq_len, 1, b
  return attention_mask
end

function create_causal_attention_mask(embedding_output, attention_mask)
  attention_mask = maybe_prepare_causal_mask(embedding_output, attention_mask)
  batched_triu!(attention_mask, 0)
  return create_attention_mask(attention_mask)
end

Flux.@nograd create_attention_mask
Flux.@nograd create_causal_attention_mask

(bm::HGFBertModel)(input; position_ids = nothing, token_type_ids = nothing,
                   attention_mask = nothing,
                   output_attentions = false,
                   output_hidden_states = false
                   ) = bm(input, position_ids, token_type_ids, attention_mask,
                          Val(output_attentions), Val(output_hidden_states))

function (bm::HGFBertModel)(
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

(bm::HGFBertModel)(input, encoder_hidden_states;
                   position_ids = nothing, token_type_ids = nothing,
                   attention_mask = nothing, encoder_attention_mask = nothing,
                   output_attentions = false,
                   output_hidden_states = false
                   ) = bm(input, encoder_hidden_states,
                          position_ids, token_type_ids,
                          attention_mask, encoder_attention_mask,
                          Val(output_attentions), Val(output_hidden_states))

function (bm::HGFBertModel)(
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

function HGFBertModel(config::HGFBertConfig)
  embeddings = HGFBertEmbeddings(config)
  encoder = HGFBertEncoder(config)
  pooler = HGFBertPooler(config)
  HGFBertModel(embeddings, encoder, pooler)
end

get_input_embedding(model::HGFBertModel) = model.embeddings.word_embeddings.weight

# bert models for different task

# pretrain

struct HGFBertForPreTraining{B<:HGFBertModel, C<:HGFBertPreTrainingHeads} <: HGFBertPreTrainedModel
  bert::B
  cls::C
end

@functor HGFBertForPreTraining

(self::HGFBertForPreTraining)(input;
                              position_ids = nothing, token_type_ids = nothing,
                              attention_mask = nothing,
                              output_attentions = false,
                              output_hidden_states = false
                              ) = self(input, position_ids, token_type_ids,
                                       attention_mask,
                                       Val(output_attentions), Val(output_hidden_states))

function (self::HGFBertForPreTraining)(input, position_ids, token_type_ids,
                                       attention_mask,
                                       _output_attentions::Val{output_attentions},
                                       _output_hidden_states::Val{output_hidden_states}
                                       ) where {output_attentions, output_hidden_states}
  outputs = self.bert(input, position_ids, token_type_ids,
                      attention_mask, _output_attentions, _output_hidden_states)
  sequence_output, pooled_output = outputs
  prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
  total_loss = nothing

  return (
    loss = total_loss,
    prediction_logits = prediction_scores,
    seq_relationship_logits = seq_relationship_score,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

(self::HGFBertForPreTraining)(input, labels, next_sentence_label;
                              position_ids = nothing, token_type_ids = nothing,
                              attention_mask = nothing,
                              output_attentions = false,
                              output_hidden_states = false
                              ) = self(input, labels, next_sentence_label,
                                       position_ids, token_type_ids,
                                       attention_mask,
                                       Val(output_attentions), Val(output_hidden_states))

function (self::HGFBertForPreTraining)(input, labels, next_sentence_label,
                                       position_ids, token_type_ids,
                                       attention_mask,
                                       _output_attentions::Val{output_attentions},
                                       _output_hidden_states::Val{output_hidden_states}
                                       ) where {output_attentions, output_hidden_states}
  outputs = self(input, position_ids, token_type_ids, attention_mask,
                _output_attentions, _output_hidden_states)

  prediction_scores = outputs.prediction_logits
  seq_relationship_score = outputs.seq_relationship_logits

  masked_lm_loss = Flux.logitcrossentropy(prediction_scores, labels)
  next_sentence_loss = Flux.logitcrossentropy(seq_relationship_score, next_sentence_label)
  total_loss = masked_lm_loss + next_sentence_loss

  return (
    loss = total_loss,
    prediction_logits = prediction_scores,
    seq_relationship_logits = seq_relationship_score,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

function HGFBertForPreTraining(config::HGFBertConfig)
  bert = HGFBertModel(config)
  input_embedding = get_input_embedding(bert)
  cls = HGFBertPreTrainingHeads(config; input_embedding=input_embedding)
  return HGFBertForPreTraining(bert, cls)
end

# clm finetune

struct HGFBertLMHeadModel{B<:HGFBertModel, C<:HGFBertOnlyMLMHead} <: HGFBertPreTrainedModel
  bert::B
  cls::C
end

@functor HGFBertLMHeadModel

(self::HGFBertLMHeadModel)(input;
                              position_ids = nothing, token_type_ids = nothing,
                              attention_mask = nothing,
                              output_attentions = false,
                              output_hidden_states = false
                              ) = self(input, position_ids, token_type_ids,
                                       attention_mask,
                                       Val(output_attentions), Val(output_hidden_states))

function (self::HGFBertLMHeadModel)(input, position_ids, token_type_ids,
                                    attention_mask,
                                    _output_attentions::Val{output_attentions},
                                    _output_hidden_states::Val{output_hidden_states}
                                    ) where {output_attentions, output_hidden_states}
  outputs = self.bert(input, position_ids, token_type_ids,
                      attention_mask, _output_attentions, _output_hidden_states)
  sequence_output = outputs[1]
  prediction_scores = self.cls(sequence_output)
  lm_loss = nothing

  return (
    loss = lm_loss,
    prediction_logits = prediction_scores,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

(self::HGFBertLMHeadModel)(input, labels;
                           position_ids = nothing, token_type_ids = nothing,
                           attention_mask = nothing,
                           output_attentions = false,
                           output_hidden_states = false
                           ) = self(input, labels,
                                    position_ids, token_type_ids,
                                    attention_mask,
                                    Val(output_attentions), Val(output_hidden_states))

function (self::HGFBertLMHeadModel)(input, labels,
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
    prediction_logits = prediction_scores,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

function HGFBertLMHeadModel(config::HGFBertConfig)
  bert = HGFBertModel(config)
  input_embedding = get_input_embedding(bert)
  cls = HGFBertOnlyMLMHead(config; input_embedding=input_embedding)
  return HGFBertLMHeadModel(bert, cls)
end

# maked lm

struct HGFBertForMaskedLM{B<:HGFBertModel, C<:HGFBertOnlyMLMHead} <: HGFBertPreTrainedModel
  bert::B
  cls::C
end

@functor HGFBertForMaskedLM

(self::HGFBertForMaskedLM)(input;
                              position_ids = nothing, token_type_ids = nothing,
                              attention_mask = nothing,
                              output_attentions = false,
                              output_hidden_states = false
                              ) = self(input, position_ids, token_type_ids,
                                       attention_mask,
                                       Val(output_attentions), Val(output_hidden_states))

function (self::HGFBertForMaskedLM)(input, position_ids, token_type_ids,
                                    attention_mask,
                                    _output_attentions::Val{output_attentions},
                                    _output_hidden_states::Val{output_hidden_states}
                                    ) where {output_attentions, output_hidden_states}
  outputs = self.bert(input, position_ids, token_type_ids,
                      attention_mask, _output_attentions, _output_hidden_states)
  sequence_output = outputs[1]
  prediction_scores = self.cls(sequence_output)
  masked_lm_loss = nothing

  return (
    loss = masked_lm_loss,
    logits = prediction_scores,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

(self::HGFBertForMaskedLM)(input, labels;
                           position_ids = nothing, token_type_ids = nothing,
                           attention_mask = nothing,
                           output_attentions = false,
                           output_hidden_states = false
                           ) = self(input, labels,
                                    position_ids, token_type_ids,
                                    attention_mask,
                                    Val(output_attentions), Val(output_hidden_states))

function (self::HGFBertForMaskedLM)(input, labels,
                                    position_ids, token_type_ids,
                                       attention_mask,
                                       _output_attentions::Val{output_attentions},
                                       _output_hidden_states::Val{output_hidden_states}
                                       ) where {output_attentions, output_hidden_states}
  outputs = self(input, position_ids, token_type_ids, attention_mask,
                _output_attentions, _output_hidden_states)

  prediction_scores = outputs.logits
  masked_lm_loss = Flux.logitcrossentropy(prediction_scores, labels)

  return (
    loss = masked_lm_loss,
    logits = prediction_scores,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

function HGFBertForMaskedLM(config::HGFBertConfig)
  bert = HGFBertModel(config)
  input_embedding = get_input_embedding(bert)
  cls = HGFBertOnlyMLMHead(config; input_embedding=input_embedding)
  return HGFBertForMaskedLM(bert, cls)
end

# next sentence

struct HGFBertForNextSentencePrediction{B<:HGFBertModel, C<:HGFBertOnlyNSPHead} <: HGFBertPreTrainedModel
  bert::B
  cls::C
end

@functor HGFBertForNextSentencePrediction

(self::HGFBertForNextSentencePrediction)(input;
                              position_ids = nothing, token_type_ids = nothing,
                              attention_mask = nothing,
                              output_attentions = false,
                              output_hidden_states = false
                              ) = self(input, position_ids, token_type_ids,
                                       attention_mask,
                                       Val(output_attentions), Val(output_hidden_states))

function (self::HGFBertForNextSentencePrediction)(input, position_ids, token_type_ids,
                                    attention_mask,
                                    _output_attentions::Val{output_attentions},
                                    _output_hidden_states::Val{output_hidden_states}
                                    ) where {output_attentions, output_hidden_states}
  outputs = self.bert(input, position_ids, token_type_ids,
                      attention_mask, _output_attentions, _output_hidden_states)
  pooled_output = outputs[2]
  seq_relationship_scores = self.cls(pooled_output)
  next_sentence_loss = nothing

  return (
    loss = next_sentence_loss,
    logits = seq_relationship_scores,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

(self::HGFBertForNextSentencePrediction)(input, next_sentence_label;
                           position_ids = nothing, token_type_ids = nothing,
                           attention_mask = nothing,
                           output_attentions = false,
                           output_hidden_states = false
                           ) = self(input, next_sentence_label,
                                    position_ids, token_type_ids,
                                    attention_mask,
                                    Val(output_attentions), Val(output_hidden_states))

function (self::HGFBertForNextSentencePrediction)(input, next_sentence_label,
                                    position_ids, token_type_ids,
                                       attention_mask,
                                       _output_attentions::Val{output_attentions},
                                       _output_hidden_states::Val{output_hidden_states}
                                       ) where {output_attentions, output_hidden_states}
  outputs = self(input, position_ids, token_type_ids, attention_mask,
                _output_attentions, _output_hidden_states)

  seq_relationship_scores = outputs.logits
  next_sentence_loss = Flux.logitcrossentropy(seq_relationship_scores, next_sentence_label)

  return (
    loss = next_sentence_loss,
    logits = seq_relationship_scores,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

function HGFBertForNextSentencePrediction(config::HGFBertConfig)
  bert = HGFBertModel(config)
  cls = HGFBertOnlyNSPHead(config)
  return HGFBertForNextSentencePrediction(bert, cls)
end

# seq classify

struct HGFBertForSequenceClassification{B<:HGFBertModel, C<:FakeTHLinear} <: HGFBertPreTrainedModel
  bert::B
  classifier::C
end

@functor HGFBertForSequenceClassification

(self::HGFBertForSequenceClassification)(input;
                              position_ids = nothing, token_type_ids = nothing,
                              attention_mask = nothing,
                              output_attentions = false,
                              output_hidden_states = false
                              ) = self(input, position_ids, token_type_ids,
                                       attention_mask,
                                       Val(output_attentions), Val(output_hidden_states))

function (self::HGFBertForSequenceClassification)(input, position_ids, token_type_ids,
                                    attention_mask,
                                    _output_attentions::Val{output_attentions},
                                    _output_hidden_states::Val{output_hidden_states}
                                    ) where {output_attentions, output_hidden_states}
  outputs = self.bert(input, position_ids, token_type_ids,
                      attention_mask, _output_attentions, _output_hidden_states)
  pooled_output = outputs[2]
  logits = self.classifier(pooled_output)
  loss = nothing

  return (
    loss = loss,
    logits = logits,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

(self::HGFBertForSequenceClassification)(input, labels;
                           position_ids = nothing, token_type_ids = nothing,
                           attention_mask = nothing,
                           output_attentions = false,
                           output_hidden_states = false
                           ) = self(input, labels,
                                    position_ids, token_type_ids,
                                    attention_mask,
                                    Val(output_attentions), Val(output_hidden_states))

function (self::HGFBertForSequenceClassification)(input, labels,
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

function HGFBertForSequenceClassification(config::HGFBertConfig)
  bert = HGFBertModel(config)
  classifier = FakeTHLinear(config, config.hidden_size, config.num_labels)
  HGFBertForSequenceClassification(bert, classifier)
end

# multiple choice

struct HGFBertForMultipleChoice{B<:HGFBertModel, C<:FakeTHLinear} <: HGFBertPreTrainedModel
  bert::B
  classifier::C
end

@functor HGFBertForMultipleChoice

(self::HGFBertForMultipleChoice)(input;
                              position_ids = nothing, token_type_ids = nothing,
                              attention_mask = nothing,
                              output_attentions = false,
                              output_hidden_states = false
                              ) = self(input, position_ids, token_type_ids,
                                       attention_mask,
                                       Val(output_attentions), Val(output_hidden_states))

function (self::HGFBertForMultipleChoice)(input, position_ids, token_type_ids,
                                    attention_mask,
                                    _output_attentions::Val{output_attentions},
                                    _output_hidden_states::Val{output_hidden_states}
                                          ) where {output_attentions, output_hidden_states}

  num_choices = size(input, ndims(input)-1)
  flat_choice(x) = reshape(x, size(x)[1:end-2]..., :)
  flat_input = flat_choice(input)
  flat_position_ids = flat_choice(position_ids)
  flat_token_type_ids = flat_choice(token_type_ids)
  flat_attention_mask = flat_choice(attention_mask)

  outputs = self.bert(flat_input, flat_position_ids, flat_token_type_ids,
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

(self::HGFBertForMultipleChoice)(input, labels;
                           position_ids = nothing, token_type_ids = nothing,
                           attention_mask = nothing,
                           output_attentions = false,
                           output_hidden_states = false
                           ) = self(input, labels,
                                    position_ids, token_type_ids,
                                    attention_mask,
                                    Val(output_attentions), Val(output_hidden_states))

function (self::HGFBertForMultipleChoice)(input, labels,
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

function HGFBertForMultipleChoice(config::HGFBertConfig)
  bert = HGFBertModel(config)
  classifier = FakeTHLinear(config, config.hidden_size, 1)
  HGFBertForMultipleChoice(bert, classifier)
end

# token classify

struct HGFBertForTokenClassification{B<:HGFBertModel, C<:FakeTHLinear} <: HGFBertPreTrainedModel
  bert::B
  classifier::C
end

@functor HGFBertForTokenClassification

(self::HGFBertForTokenClassification)(input;
                              position_ids = nothing, token_type_ids = nothing,
                              attention_mask = nothing,
                              output_attentions = false,
                              output_hidden_states = false
                              ) = self(input, position_ids, token_type_ids,
                                       attention_mask,
                                       Val(output_attentions), Val(output_hidden_states))

function (self::HGFBertForTokenClassification)(input, position_ids, token_type_ids,
                                    attention_mask,
                                    _output_attentions::Val{output_attentions},
                                    _output_hidden_states::Val{output_hidden_states}
                                    ) where {output_attentions, output_hidden_states}
  outputs = self.bert(input, position_ids, token_type_ids,
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

(self::HGFBertForTokenClassification)(input, labels;
                           position_ids = nothing, token_type_ids = nothing,
                           attention_mask = nothing,
                           output_attentions = false,
                           output_hidden_states = false
                           ) = self(input, labels,
                                    position_ids, token_type_ids,
                                    attention_mask,
                                    Val(output_attentions), Val(output_hidden_states))

function (self::HGFBertForTokenClassification)(input, labels,
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

function HGFBertForTokenClassification(config::HGFBertConfig)
  bert = HGFBertModel(config)
  classifier = FakeTHLinear(config, config.hidden_size, config.num_labels)
  HGFBertForTokenClassification(bert, classifier)
end

# qa

struct HGFBertForQuestionAnswering{B<:HGFBertModel, C<:FakeTHLinear} <: HGFBertPreTrainedModel
  bert::B
  qa_outputs::C
end

@functor HGFBertForQuestionAnswering

(self::HGFBertForQuestionAnswering)(input;
                              position_ids = nothing, token_type_ids = nothing,
                              attention_mask = nothing,
                              output_attentions = false,
                              output_hidden_states = false
                              ) = self(input, position_ids, token_type_ids,
                                       attention_mask,
                                       Val(output_attentions), Val(output_hidden_states))

function (self::HGFBertForQuestionAnswering)(input, position_ids, token_type_ids,
                                    attention_mask,
                                    _output_attentions::Val{output_attentions},
                                    _output_hidden_states::Val{output_hidden_states}
                                    ) where {output_attentions, output_hidden_states}
  outputs = self.bert(input, position_ids, token_type_ids,
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

(self::HGFBertForQuestionAnswering)(input, start_positions, end_positions;
                           position_ids = nothing, token_type_ids = nothing,
                           attention_mask = nothing,
                           output_attentions = false,
                           output_hidden_states = false
                           ) = self(input, start_positions, end_positions,
                                    position_ids, token_type_ids,
                                    attention_mask,
                                    Val(output_attentions), Val(output_hidden_states))

function (self::HGFBertForQuestionAnswering)(input, start_positions, end_positions,
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

function HGFBertForQuestionAnswering(config::HGFBertConfig)
  bert = HGFBertModel(config)
  classifier = FakeTHLinear(config, config.hidden_size, config.num_labels)
  HGFBertForQuestionAnswering(bert, classifier)
end
