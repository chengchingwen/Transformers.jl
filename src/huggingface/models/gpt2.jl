using ..Transformers: batchedmul, batched_triu!

# GPT2 specific initializers

function FakeTHLinear(config::HGFGPT2Config, hi, ho; bias=true)
  weight = randn(Float32, ho, hi) .* config.initializer_range
  if bias
    bias = zeros(Float32, ho)
  else
    bias = nothing
  end
  FakeTHLinear(weight, bias)
end

function FakeTHEmbedding(config::HGFGPT2Config, num, dims)
  weight = randn(Float32, dims, num) .* config.initializer_range
  FakeTHEmbedding(0, weight)
end

function FakeTHLayerNorm(config::HGFGPT2Config, dims; eps::Float32=1e-05)
  weight = ones(Float32, dims)
  bias = zeros(Float32, dims)
  FakeTHLayerNorm(eps, weight, bias)
end

function FakeHGFConv1D(config::HGFGPT2Config, nf, nx)
  weight = randn(Float32, nx, nf) .* config.initializer_range
  bias = zeros(Float32, nf)
  FakeHGFConv1D(weight, bias)
end

function FakeHGFSequenceSummary(config::HGFGPT2Config)
  global ACT2FN
  summary_type = SummaryType(Symbol(config.summary_type))
  if config.summary_use_proj
    num_classes = config.num_labels
  else
    num_classes = config.n_embd
  end
  summary = FakeTHLinear(config, config.n_embd, num_classes)
  if isnothing(config.summary_activation)
    activation = identity
  else
    activation = ACT2FN[Symbol(config.summary_activation)]
  end
  return FakeHGFSequenceSummary{summary_type}(summary, activation)
end

# HGF GPT2 compounts

# attention

struct HGFGPT2Attention{P<:FakeHGFConv1D, O<:FakeHGFConv1D} <: THModule
  num_attention_heads::Int
  c_attn::P
  c_proj::O
end

Functors.functor(::Type{<:HGFGPT2Attention}, a) = (c_attn=a.c_attn, c_proj=a.c_proj), y->HGFGPT2Attention(a.num_attention_heads, y...)

function _split_qkv(x)
  h = size(x, 1)
  nh = div(h, 3)
  return (
    @view(x[1:nh, :, :]),
    @view(x[nh+1:2nh, :, :]),
    @view(x[2nh+1:3nh, :, :]),
  )
end

function (self::HGFGPT2Attention)(x, attention_mask,
                                  _output_attentions::Val{output_attentions},
                                  _use_cache::Val{use_cache}
                                  ) where {output_attentions, use_cache}
  x = self.c_attn(x)
  query, key, value = _split_qkv(x)
  query = _split_tranpose_for_scores(query, self.num_attention_heads)
  key = _split_tranpose_for_scores(key, self.num_attention_heads)
  value = _split_tranpose_for_scores(value, self.num_attention_heads)
  self(query, key, value, attention_mask, _output_attentions, _use_cache)
end

function (self::HGFGPT2Attention)(x, past, attention_mask,
                                  _output_attentions::Val{output_attentions},
                                  _use_cache::Val{use_cache}
                                  ) where {output_attentions, use_cache}
  x = self.c_attn(x)
  query, key, value = _split_qkv(x)
  query = _split_tranpose_for_scores(query, self.num_attention_heads)
  key = _split_tranpose_for_scores(key, self.num_attention_heads)
  value = _split_tranpose_for_scores(value, self.num_attention_heads)

  past_key, past_value = past
  key = hcat(past_key, key)
  value = hcat(past_value, value)

  self(query, key, value, attention_mask, _output_attentions, _use_cache)
end


struct ShiftAttentionMask{T}
  mask::T
  past_length::Int
end

function apply_shift_mask(scores, mask)
    out = copy(scores)
    @view(out[1:(end-mask.past_length), :, :, :]) .+= mask.mask
    return out
end

@adjoint function apply_shift_mask(scores, mask)
    out = apply_shift_mask
    return out, Δ -> (Δ, nothing)
end

function _compute_attention_scores(query, key, attention_mask::ShiftAttentionMask)
  w_wo_mask = _compute_attention_scores(query, key, nothing)
  return apply_shift_mask(w_wo_mask, attention_mask)
end

function _attn(query, key, value, attention_mask)
  w = _compute_attention_scores(query, key, attention_mask)
  w = softmax(w; dims=1)
  a = batchedmul(value, w)
  return a, w
end

function (self::HGFGPT2Attention)(query, key, value, attention_mask,
                                  ::Val{output_attentions},
                                  ::Val{use_cache}
                                  ) where {output_attentions, use_cache}

  if use_cache
    present = (key, value)
  end
    
  a, w = _attn(query, key, value, attention_mask)
  a = _merge_transpose_for_output(a, self.num_attention_heads)
  a = self.c_proj(a)

  if output_attentions && use_cache
    return a, present, w
  elseif output_attentions
    return a, w
  elseif use_cache
    return a, present
  else
    return a
  end
end

function HGFGPT2Attention(config::HGFGPT2Config, nx, n_ctx)
  @assert nx % config.n_head == 0
  c_attn = FakeHGFConv1D(config, 3nx, nx)
  c_proj = FakeHGFConv1D(config, nx, nx)
  return HGFGPT2Attention(config.n_head, c_attn, c_proj)
end

# positionwise

struct HGFGPT2MLP{F, P1<:FakeHGFConv1D, P2<:FakeHGFConv1D} <: THModule
  act::F
  c_fc::P1
  c_proj::P2
end

Functors.functor(::Type{<:HGFGPT2MLP}, mlp) = (c_fc = mlp.c_fc, c_proj = mlp.c_proj), y->HGFGPT2MLP(mlp.act, y...)

(mlp::HGFGPT2MLP)(x) = mlp.c_proj(mlp.act.(mlp.c_fc(x)))

function HGFGPT2MLP(config::HGFGPT2Config, n_state)
  global ACT2FN
  act = ACT2FN[Symbol(config.activation_function)]
  c_fc = FakeHGFConv1D(config, n_state, config.n_embd)
  c_proj = FakeHGFConv1D(config, config.n_embd, n_state)
  return HGFGPT2MLP(act, c_fc, c_proj)
end

# pre-ln transformer block

struct HGFGPT2Block{A<:HGFGPT2Attention, L1<:FakeTHLayerNorm, L2<:FakeTHLayerNorm, M<:HGFGPT2MLP} <: THModule
  ln_1::L1
  attn::A
  ln_2::L2
  mlp::M
end

@functor HGFGPT2Block

function (self::HGFGPT2Block)(x, attention_mask,
                              _output_attentions::Val{output_attentions},
                              _use_cache::Val{use_cache}) where {output_attentions, use_cache}
  _output_attn = self.attn(self.ln_1(x), attention_mask, _output_attentions, _use_cache)

  if output_attentions && use_cache
    output_attn, present, attention_prob = _output_attn
  elseif output_attentions
    output_attn, attention_prob = _output_attn
  elseif use_cache
    output_attn, present = _output_attn
  else
    output_attn = _output_attn
  end

  x = x + output_attn
  m = self.mlp(self.ln_2(x))
  x = x + m

  if output_attentions && use_cache
    return x, present, attention_prob
  elseif output_attentions
    return x, attention_prob
  elseif use_cache
    return x, present
  else
    return x
  end
end

function (self::HGFGPT2Block)(x, past, attention_mask,
                              _output_attentions::Val{output_attentions},
                              _use_cache::Val{use_cache}) where {output_attentions, use_cache}
  _output_attn = self.attn(self.ln_1(x), past, attention_mask, _output_attentions, _use_cache)

  if output_attentions && use_cache
    output_attn, present, attention_prob = _output_attn
  elseif output_attentions
    output_attn, attention_prob = _output_attn
  elseif use_cache
    output_attn, present = _output_attn
  else
    output_attn = _output_attn
  end

  x = x + output_attn
  m = self.mlp(self.ln_2(x))
  x = x + m

  if output_attentions && use_cache
    return x, present, attention_prob
  elseif output_attentions
    return x, attention_prob
  elseif use_cache
    return x, present
  else
    return x
  end
end

function HGFGPT2Block(config::HGFGPT2Config, n_ctx)
  nx = config.n_embd
  ln_1 = FakeTHLayerNorm(config, nx; eps=config.layer_norm_epsilon)
  attn = HGFGPT2Attention(config, nx, n_ctx)
  ln_2 = FakeTHLayerNorm(config, nx; eps=config.layer_norm_epsilon)
  mlp = HGFGPT2MLP(config, 4nx)
  return HGFGPT2Block(ln_1, attn, ln_2, mlp)
end

# gpt2 model

abstract type HGFGPT2PreTrainedModel <: HGFPreTrainedModel end

struct HGFGPT2Model{N, L<:FakeTHModuleList{N},
                    WE<:FakeTHEmbedding,
                    PE<:FakeTHEmbedding,
                    LN<:FakeTHLayerNorm
                    } <: HGFGPT2PreTrainedModel
  wte::WE
  wpe::PE
  h::L
  ln_f::LN
end

@functor HGFGPT2Model

@inline get_past_length(past_key_values) = size(past_key_values[1][1], 2)
@inline get_past_length(::Nothing) = 0

@inline get_word_emb(word_embeddings::FakeTHEmbedding, input_ids::AbstractArray{<:Integer}) = word_embeddings(input_ids)
@inline get_word_emb(word_embeddings::FakeTHEmbedding, input_embed::AbstractArray{T}) where T = input_embed

@inline maybe_past_arange(inputs_embeds, ::Nothing) = _arange(inputs_embeds, size(inputs_embeds)[end-1])
@inline maybe_past_arange(inputs_embeds, past_key_values) = _arange(inputs_embeds, get_past_length(past_key_values), size(inputs_embeds)[end-1])

@inline get_position_emb(position_embeddings::FakeTHEmbedding, inputs_embeds, ::Nothing, past_key_values) = get_position_emb(position_embeddings, inputs_embeds, maybe_past_arange(inputs_embeds, past_key_values), past_key_values)
@inline get_position_emb(position_embeddings::FakeTHEmbedding, inputs_embeds, position_ids, past_key_values) = position_embeddings(position_ids)

@inline get_token_type_emb(token_type_embeddings::FakeTHEmbedding, token_type_ids) = token_type_embeddings(token_type_ids)

@inline apply_token_type_emb(token_type_embeddings::FakeTHEmbedding, embeddings, ::Nothing) = embeddings
@inline apply_token_type_emb(token_type_embeddings::FakeTHEmbedding, embeddings, token_type_ids) = embeddings + get_token_type_emb(token_type_embeddings, token_type_ids)

function (self::HGFGPT2Model)(input, position_ids, token_type_ids, past_key_values)
  inputs_embeds = get_word_emb(self.wte, input)
  position_embeds = get_position_emb(self.wpe, inputs_embeds, position_ids, past_key_values)
  embeddings = inputs_embeds .+ position_embeds
  embeddings = apply_token_type_emb(self.wte, embeddings, token_type_ids)
  return embeddings
end

function (self::HGFGPT2Model)(input; position_ids=nothing, token_type_ids=nothing,
                              past_key_values=nothing,
                              attention_mask=nothing,
                              output_attentions=false,
                              output_hidden_states=false,
                              use_cache=false)
  return self(input, position_ids, token_type_ids,
              past_key_values, attention_mask,
              Val(output_attentions),
              Val(output_hidden_states),
              Val(use_cache))
end

function (self::HGFGPT2Model)(input, position_ids, token_type_ids, attention_mask,
                              _output_attentions::Val,
                              _output_hidden_states::Val,
                              _use_cache::Val
                              )
  return self(input, position_ids, token_type_ids, nothing, attention_mask,
              _output_attentions,
              _output_hidden_states,
              _use_cache)
end

@generated function (self::HGFGPT2Model{N})(input, position_ids, token_type_ids, past, attention_mask,
                                            _output_attentions::Val{output_attentions},
                                            _output_hidden_states::Val{output_hidden_states},
                                            _use_cache::Val{use_cache}
                                            ) where {N, output_attentions, output_hidden_states, use_cache}

  if output_attentions
    all_attentions = Expr(:tuple)
  end

  if output_hidden_states
    all_hidden_states = Expr(:tuple, :hidden_state)
  end

  if use_cache
    presents = Expr(:tuple)
  end

  body = Expr[]

  for i = 1:N
    previous = i == 1 ? :hidden_state : Symbol(:hidden_state, i-1)
    current = Symbol(:hidden_state, i)

    if output_attentions && use_cache
      current_attention = Symbol(:attention_, i)
      current_present = Symbol(:present_, i)
      current_output = :($current, $current_present, $current_attention)
    elseif output_attentions
      current_attention = Symbol(:attention_, i)
      current_output = :($current, $current_attention)
    elseif use_cache
      current_present = Symbol(:present_, i)
      current_output = :($current, $current_present)
    else
      current_output = current
    end

    if past <: Nothing
      expr = :($current_output = self.h[$i]($previous, attention_mask, _output_attentions, _use_cache))
    else
      expr = :($current_output = self.h[$i]($previous, past[$i], attention_mask, _output_attentions, _use_cache))
    end
    push!(body, expr)

    if output_attentions
      push!(all_attentions.args, current_attention)
    end

    if output_hidden_states
      push!(all_hidden_states.args, current)
    end

    if use_cache
      push!(presents.args, current_present)
    end
  end

  current = Symbol(:hidden_state, N)
  push!(body, :($current = self.ln_f($current)))

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

  if use_cache
    push!(body, :(presents = $presents))
  else
    push!(body, :(presents = nothing))
  end
    
  mask_expr = [:(attention_mask = create_causal_attention_mask(hidden_state, attention_mask))]
  if !(past <: Nothing)
    push!(mask_expr, :(attention_mask = ShiftAttentionMask(attention_mask, get_past_length(past))))
  end
    
  return quote
    hidden_state = self(input, position_ids, token_type_ids, past)
    $(mask_expr...)
    $(body...)

    return (
      last_hidden_state = $current,
      past_key_values = presents,
      hidden_states = all_hidden_states,
      attentions = all_attentions
    )
  end
end

function HGFGPT2Model(config::HGFGPT2Config)
  wte = FakeTHEmbedding(config, config.vocab_size, config.n_embd)
  wpe = FakeTHEmbedding(config, config.n_positions, config.n_embd)
  h = FakeTHModuleList(
    [HGFGPT2Block(config, config.n_ctx) for _ in 1:config.n_layer]
  )
  ln_f = FakeTHLayerNorm(config, config.n_embd; eps=config.layer_norm_epsilon)
  return HGFGPT2Model(wte, wpe, h, ln_f)
end

#

struct HGFGPT2LMHeadModel{T<:HGFGPT2Model, L<:FakeTHLinear} <: HGFGPT2PreTrainedModel
  transformer::T
  lm_head::L
end

@functor HGFGPT2LMHeadModel

function (self::HGFGPT2LMHeadModel)(input; position_ids=nothing, token_type_ids=nothing,
                                    past_key_values=nothing,
                                    attention_mask=nothing,
                                    output_attentions=false,
                                    output_hidden_states=false,
                                    use_cache=false)
  return self(input, position_ids, token_type_ids,
              past_key_values, attention_mask,
              Val(output_attentions),
              Val(output_hidden_states),
              Val(use_cache))
end

function (self::HGFGPT2LMHeadModel)(input, position_ids, token_type_ids, attention_mask,
                                    _output_attentions::Val,
                                    _output_hidden_states::Val,
                                    _use_cache::Val
                                    )
  return self(input, position_ids, token_type_ids,
              nothing, attention_mask,
              _output_attentions,
              _output_hidden_states,
              _use_cache)
end

function (self::HGFGPT2LMHeadModel)(input, position_ids, token_type_ids, past::Union{Tuple, Nothing}, attention_mask,
                                    _output_attentions::Val,
                                    _output_hidden_states::Val,
                                    _use_cache::Val
                                    )
  transformer_outputs = self.transformer(input, position_ids, token_type_ids, past, attention_mask,
                                         _output_attentions,
                                         _output_hidden_states,
                                         _use_cache)
  hidden_states = transformer_outputs.last_hidden_state
  lm_logits = self.lm_head(hidden_states)
  loss = nothing

  return (
    loss = loss,
    logits = lm_logits,
    past_key_values = transformer_outputs.past_key_values,
    hidden_states = transformer_outputs.hidden_states,
    attentions = transformer_outputs.attentions
  )
end

function (self::HGFGPT2LMHeadModel)(input, labels; position_ids=nothing, token_type_ids=nothing,
                                    past_key_values=nothing,
                                    attention_mask=nothing,
                                    output_attentions=false,
                                    output_hidden_states=false,
                                    use_cache=false)
  return self(input, labels, position_ids, token_type_ids,
              past_key_values, attention_mask,
              Val(output_attentions),
              Val(output_hidden_states),
              Val(use_cache))
end

function (self::HGFGPT2LMHeadModel)(input, labels, position_ids, token_type_ids, attention_mask,
                                    _output_attentions::Val,
                                    _output_hidden_states::Val,
                                    _use_cache::Val
                                    )
  return self(input, labels, position_ids, token_type_ids,
              nothing, attention_mask,
              _output_attentions,
              _output_hidden_states,
              _use_cache)
end

function (self::HGFGPT2LMHeadModel)(input, labels, position_ids, token_type_ids,
                                    past::Union{Tuple, Nothing}, attention_mask,
                                    _output_attentions::Val,
                                    _output_hidden_states::Val,
                                    _use_cache::Val
                                    )
  outputs = self(input, position_ids, token_type_ids, past, attention_mask,
                 _output_attentions,
                 _output_hidden_states,
                 _use_cache)
  shift_logits = outputs.logits[:, 1:end-1, :]
  shift_labels = labels[:, 2:end, :]
  loss = Flux.logitcrossentropy(shift_logits, shift_labels)

  return (
    loss = loss,
    logits = outputs.logits,
    past_key_values = outputs.past_key_values,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

function HGFGPT2LMHeadModel(config::HGFGPT2Config)
  transformer = HGFGPT2Model(config)
  lm_head = FakeTHLinear(config, config.n_embd, config.vocab_size; bias=false)
  return HGFGPT2LMHeadModel(transformer, lm_head)
end

#

struct HGFGPT2DoubleHeadsModel{T<:HGFGPT2Model, L<:FakeTHLinear, S<:FakeHGFSequenceSummary} <: HGFGPT2PreTrainedModel
  transformer::T
  lm_head::L
  multiple_choice_head::S
end

@functor HGFGPT2DoubleHeadsModel

function (self::HGFGPT2DoubleHeadsModel)(input; position_ids=nothing, token_type_ids=nothing,
                                          past_key_values=nothing,
                                          mc_token_ids=nothing,
                                          attention_mask=nothing,
                                          output_attentions=false,
                                          output_hidden_states=false,
                                          use_cache=false)
  return self(input, position_ids, token_type_ids,
              past_key_values, mc_token_ids, attention_mask,
              Val(output_attentions),
              Val(output_hidden_states),
              Val(use_cache))
end

function (self::HGFGPT2DoubleHeadsModel)(input, position_ids, token_type_ids, mc_token_ids, attention_mask,
                                    _output_attentions::Val,
                                    _output_hidden_states::Val,
                                    _use_cache::Val
                                    )
  return self(input, position_ids, token_type_ids,
              nothing, mc_token_ids, attention_mask,
              _output_attentions,
              _output_hidden_states,
              _use_cache)
end

function (self::HGFGPT2DoubleHeadsModel)(input, position_ids, token_type_ids, past, mc_token_ids, attention_mask,
                                    _output_attentions::Val,
                                    _output_hidden_states::Val,
                                    _use_cache::Val
                                         )

  num_choices = size(input, ndims(input)-1)
  flat_choice(x) = reshape(x, size(x)[1:end-2]..., :)
  flat_choice(::Nothing) = nothing
  map_flat_choice2(x) = map(x) do (k, v)
    (flat_choice(k), flat_choice(v))
  end
  map_flat_choice2(::Nothing) = nothing

  flat_input = flat_choice(input)
  flat_position_ids = flat_choice(position_ids)
  flat_token_type_ids = flat_choice(token_type_ids)
  flat_attention_mask = flat_choice(attention_mask)
  flat_past = map_flat_choice2(past)

  transformer_outputs = self.transformer(flat_input, flat_position_ids, flat_token_type_ids,
                                         flat_past, flat_attention_mask,
                                         _output_attentions,
                                         _output_hidden_states,
                                         _use_cache)
  hidden_states = transformer_outputs.last_hidden_state |> Base.Fix2(reshape, (num_choices, :))
  lm_logits = self.lm_head(hidden_states)
  mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids)
  lm_loss = nothing
  mc_loss = nothing

  return (
    lm_loss = lm_loss,
    mc_loss = mc_loss,
    lm_logits = lm_logits,
    mc_logits = mc_logits,
    past_key_values = transformer_outputs.past_key_values,
    hidden_states = transformer_outputs.hidden_states,
    attentions = transformer_outputs.attentions
  )
end

function (self::HGFGPT2DoubleHeadsModel)(input, labels, mc_labels; position_ids=nothing, token_type_ids=nothing,
                                          past_key_values=nothing, mc_token_ids=nothing,
                                          attention_mask=nothing,
                                          output_attentions=false,
                                          output_hidden_states=false,
                                          use_cache=false)
  return self(input, labels, mc_labels, position_ids, token_type_ids,
              past_key_values, mc_token_ids, attention_mask,
              Val(output_attentions),
              Val(output_hidden_states),
              Val(use_cache))
end

function (self::HGFGPT2DoubleHeadsModel)(input, labels, mc_labels, position_ids, token_type_ids,
                                          mc_token_ids, attention_mask,
                                          _output_attentions::Val,
                                          _output_hidden_states::Val,
                                          _use_cache::Val
                                          )
  return self(input, labels, mc_labels, position_ids, token_type_ids,
              nothing, mc_token_ids, attention_mask,
              _output_attentions,
              _output_hidden_states,
              _use_cache)
end

function (self::HGFGPT2DoubleHeadsModel)(input, labels, mc_labels, position_ids, token_type_ids,
                                          past, mc_token_ids, attention_mask,
                                          _output_attentions::Val,
                                          _output_hidden_states::Val,
                                          _use_cache::Val
                                          )
  outputs = self(input, position_ids, token_type_ids, past, attention_mask,
                 _output_attentions,
                 _output_hidden_states,
                 _use_cache)
  shift_logits = outputs.lm_logits[:, 1:end-1, :, :]
  shift_labels = labels[:, 2:end, :, :]
  lm_loss = Flux.logitcrossentropy(shift_logits, shift_labels)

  mc_loss = Flux.logitcrossentropy(outputs.mc_logits, mc_labels)

  return (
    lm_loss = lm_loss,
    mc_loss = mc_loss,
    lm_logits = outputs.lm_logits,
    mc_logits = outputs.mc_logits,
    past_key_values = outputs.past_key_values,
    hidden_states = outputs.hidden_states,
    attentions = outputs.attentions
  )
end

function HGFGPT2DoubleHeadsModel(config::HGFGPT2Config)
  transformer = HGFGPT2Model(config)
  lm_head = FakeTHLinear(config, config.n_embd, config.vocab_size; bias=false)
  multiple_choice_head = FakeHGFSequenceSummary(config)
  return HGFGPT2DoubleHeadsModel(transformer, lm_head, multiple_choice_head)
end

# load model utils

basemodelkey(::HGFGPT2PreTrainedModel) = :transformer
basemodel(m::HGFGPT2PreTrainedModel) = getproperty(m, basemodelkey(m))
basemodel(m::HGFGPT2Model) = m

isbasemodel(m::HGFGPT2Model) = true
isbasemodel(m::HGFGPT2PreTrainedModel) = false

get_model_type(::Val{:gpt2}) = (
  :model => HGFGPT2Model,
  :lmheadmodel => HGFGPT2LMHeadModel,
  :doubleheadsmodel => HGFGPT2DoubleHeadsModel,
)
