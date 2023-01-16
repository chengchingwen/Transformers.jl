using ..Transformers
using NeuralAttentionlib: score_returning, multihead_qkv_attention, CausalMask


function FakeTHEmbedding(::Type{HGFCLIPTextConfig}, config, num, dims; pad_idx = nothing)
    weight = randn(Float32, dims, num) .* config.initializer_range
    if !isnothing(pad_idx)
        real_pad_idx = pad_idx + 1
    else
        real_pad_idx = 0
    end
    FakeTHEmbedding(real_pad_idx, weight)
end

function FakeTHLinear(::Type{HGFCLIPTextConfig}, config, hi, ho; bias=true)
  weight = randn(Float32, ho, hi) .* config.initializer_range
  if bias
    bias = zeros(Float32, ho)
  else
    bias = nothing
  end
  FakeTHLinear(weight, bias)
end

function FakeTHLayerNorm(::Type{HGFCLIPTextConfig}, config, dims; eps::Float32=1e-05)
    weight = ones(Float32, dims)
    bias = zeros(Float32, dims)
    FakeTHLayerNorm(eps, weight, bias)
end


# Embedding
struct HGFCLIPTextEmbeddings{P<:FakeTHEmbedding,T<:FakeTHEmbedding} <: THModule
    position_embedding::P
    token_embedding::T
end

@functor HGFCLIPTextEmbeddings

"""
    HGFCLIPTextEmbeddings(config::HGFCLIPTextConfig)
Create HGFCLIPTextEmbeddings from the HGFCLIPTextConfig.

# Example
```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> embeddings = Transformers.HuggingFace.HGFCLIPTextEmbeddings(clip_config.text_config)
julia> state = Transformers.load_state(clip_model_name)
julia> Transformers.HuggingFace.load_state!(embeddings, state.text_model.embeddings)
```
"""
HGFCLIPTextEmbeddings(cfg::AbstractHGFConfig, args...; kwargs...) = HGFCLIPTextEmbeddings(typeof(cfg), cfg, args...; kwargs...)
function HGFCLIPTextEmbeddings(T::Type{HGFCLIPTextConfig}, config) 
    position_emb = FakeTHEmbedding(T, config, config.max_position_embeddings, config.hidden_size)
    token_emb = FakeTHEmbedding(T, config, config.vocab_size, config.hidden_size)
    HGFCLIPTextEmbeddings(position_emb, token_emb)
end

@inline get_word_emb(emb::HGFCLIPTextEmbeddings, input_ids::AbstractArray{<:Integer}) = emb.token_embedding(input_ids)

"""
    get_position_emb(emb::HGFCLIPTextEmbeddings, position_ids::AbstractArray{<:Integer})
Returns the position embedding for the input `position_ids` using `emb.position_embedding` .

# Example
```julia-repl
julia> embeddings = HGFCLIPTextEmbeddings(Transformers.load_config("openai/clip-vit-large-patch14").text_config)
julia> get_position_emb(embeddings, cumsum(ones(Int32, 16)))
```
"""
@inline get_position_emb(emb::HGFCLIPTextEmbeddings, position_ids) = emb.position_embedding(position_ids)
function get_position_emb(emb::HGFCLIPTextEmbeddings, ::Nothing)
    # In HGF, position_ids are 0-based range array (1, config.max_position_embeddings)
    # we create a 1-based int array ranging from 1 to config.max_position_embeddings.  
    get_position_emb(emb, cumsum(ones(Int32, size(emb.position_embedding.weight)[2])))
end

# Forward definitions of HGFCLIPTextEmbeddings
"""
    HGFCLIPTextEmbeddings(inputs_embeds::AbstractArray{Float32}, position_embeds::AbstractArray{Float32})
Evaluate the forward of HGFCLIPTextEmbeddings given input and position embeddings.

# Example
```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> embeddings = Transformers.HuggingFace.HGFCLIPTextEmbeddings(clip_config.text_config)
julia> embeddings(randn(768,1,1), randn(768,1,1))
```
"""
function (self::HGFCLIPTextEmbeddings)(inputs_embeds::AbstractArray{T}, position_embeds::AbstractArray{T}) where T
  return (inputs_embeds .+ position_embeds)
end

"""
    HGFCLIPTextEmbeddings(input_ids::AbstractArray{<:Integer}, position_ids::AbstractArray{<:Integer})
Evaluate the forward of HGFCLIPTextEmbeddings given `input_ids` and `position_ids`.

# Example
```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> embeddings = Transformers.HuggingFace.HGFCLIPTextEmbeddings(clip_config.text_config)
julia> embeddings(ones(Int32, 77, 1), ones(Int32, 77, 1))
```
"""
function (self::HGFCLIPTextEmbeddings)(
    input_ids::AbstractArray{<:Integer},
    position_ids::Union{Nothing,AbstractArray{<:Integer}},
)
    inputs_embeds = get_word_emb(self, input_ids)
    position_embeds = get_position_emb(self, position_ids)
    return self(inputs_embeds, position_embeds)
end

"""
    HGFCLIPTextEmbeddings(input_ids::AbstractArray{<:Integer}; position_ids=nothing)
Evaluate the forward of HGFCLIPTextEmbeddings given `input_ids`.

# Example
```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> embeddings = Transformers.HuggingFace.HGFCLIPTextEmbeddings(clip_config.text_config)
julia> embeddings(ones(Int32, 77, 1))
```
"""
(self::HGFCLIPTextEmbeddings)(input_ids; position_ids=nothing) = self(input_ids, position_ids)


# HGFCLIPMLP
struct HGFCLIPMLP{F1<:FakeTHLinear, F2<:FakeTHLinear, A<:Any} <: THModule
    fc1::F1
    fc2::F2
    activation_fn::A
end

Functors.functor(::Type{<:HGFCLIPMLP}, mlp) = (fc1=mlp.fc1, fc2=mlp.fc2), y->HGFCLIPMLP(y..., mlp.activation_fn)

"""
    HGFCLIPMLP(config::HGFCLIPTextConfig)
Create HGFCLIPMLP model from HGFCLIPTextConfig config.

# Example for creating model from config
```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> clip_mlp = Transformers.HuggingFace.HGFCLIPMLP(clip_config.text_config)
```

# Example for loading CLIPMLP from a model in HuggingFace
```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> clip_mlp = Transformers.HuggingFace.HGFCLIPMLP(clip_config.text_config)
julia> state_dict = Transformers.load_state("openai/clip-vit-large-patch14")
julia> Transformers.HuggingFace.load_state!(clip_mlp, state.text_model.encoder.layers[1].mlp)
```
"""
HGFCLIPMLP(cfg::AbstractHGFConfig, args...; kwargs...) = HGFCLIPMLP(typeof(cfg), cfg, args...; kwargs...)
function HGFCLIPMLP(T::Type{HGFCLIPTextConfig}, config::HGFCLIPTextConfig)
    fc1 = FakeTHLinear(T, config, config.hidden_size, config.intermediate_size)
    fc2 = FakeTHLinear(T, config, config.intermediate_size, config.hidden_size)
    act = ACT2FN[Symbol(config.hidden_act)]
    HGFCLIPMLP(fc1, fc2, act)
end

# https://github.com/huggingface/transformers/blob/f68796bd603ef60173e093f50a1ecbb74bc2ba6b/src/transformers/models/clip/modeling_clip.py#L335
# Forward definitions of HGFCLIPMLP
"""
    HGFCLIPMLP(hidden_states::AbstractArray)
Evaluate the forward of HGFCLIPMLP given `hidden_states`. 

# Example
```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> clip_mlp = Transformers.HuggingFace.HGFCLIPMLP(clip_config.text_config)
julia> out = clip_mlp(randn(768, 1, 1))
```
"""
function (self::HGFCLIPMLP)(hidden_states::AbstractArray{T}) where T
    return self.fc2(self.activation_fn.(self.fc1(hidden_states)))
end

# TODO: Ask Peter; why do we need functors here? how are they used?
# Functors.functor(::Type{<:HGFGPT2MLP}, mlp) = (c_fc = mlp.c_fc, c_proj = mlp.c_proj), y->HGFGPT2MLP(mlp.act, y...)

# HGFCLIPAttention
struct HGFCLIPAttention <: THModule
    num_heads::Int
    dropout::Float64
    q_proj::FakeTHLinear
    k_proj::FakeTHLinear
    v_proj::FakeTHLinear
    out_proj::FakeTHLinear
end

Functors.functor(::Type{<:HGFCLIPAttention}, attn) = (q_proj=attn.q_proj, k_proj=attn.k_proj, v_proj=attn.v_proj, out_proj=attn.out_proj), y->HGFCLIPAttention(attn.num_heads, attn.dropout, y...)

"""
    HGFCLIPAttention(config::HGFCLIPTextConfig)
Create HGFCLIPAttention model from HGFCLIPTextConfig config.

# Example for creating model from config
```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> clip_attn = Transformers.HuggingFace.HGFCLIPAttention(clip_config.text_config)
```

# Example for loading HGFCLIPAttention from HuggingFace model
```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> clip_attn = Transformers.HuggingFace.HGFCLIPAttention(clip_config.text_config)
julia> state_dict = Transformers.load_state("openai/clip-vit-large-patch14")
julia> Transformers.HuggingFace.load_state!(clip_attn, state.text_model.encoder.layers[1].self_attn)
```
"""
HGFCLIPAttention(cfg::AbstractHGFConfig, args...; kwargs...) = HGFCLIPAttention(typeof(cfg), cfg, args...; kwargs...)
function HGFCLIPAttention(T::Type{HGFCLIPTextConfig}, config::HGFCLIPTextConfig)
    num_heads = config.num_attention_heads
    dropout = config.attention_dropout
    q_proj = FakeTHLinear(T, config, config.hidden_size, config.hidden_size)
    k_proj = FakeTHLinear(T, config, config.hidden_size, config.hidden_size)
    v_proj = FakeTHLinear(T, config, config.hidden_size, config.hidden_size)
    out_proj = FakeTHLinear(T, config, config.hidden_size, config.hidden_size)
    HGFCLIPAttention(num_heads, dropout, q_proj, k_proj, v_proj, out_proj)
end

# https://github.com/huggingface/transformers/blob/f68796bd603ef60173e093f50a1ecbb74bc2ba6b/src/transformers/models/clip/modeling_clip.py#L231
"""
    HGFCLIPAttention(hidden_states)
Evaluate the forward of HGFCLIPAttention given `hidden_states`. 

# An example inference of the "openai/clip-vit-large-patch14" model 
```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> clip_attn = Transformers.HuggingFace.HGFCLIPAttention(clip_config.text_config)
julia> state_dict = Transformers.load_state("openai/clip-vit-large-patch14")
julia> Transformers.HuggingFace.load_state!(clip_attn, state_dict.text_model.encoder.layers[1].self_attn)
julia> clip_attn(zero(Array{Float32, 3}(undef, 768, 1, 1)))
```

"""
function (self::HGFCLIPAttention)(
    hidden_states::AbstractArray{T};                                 # embed_dim x seq_len x batch
    attention_mask::Union{Nothing}=nothing,
    causal_attention_mask::Union{CausalMask, Nothing}=nothing,
    output_attentions::Bool=false
) where T

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # masks should be applied in this order: causal_attention_mask -> attention_mask
    masks = causal_attention_mask  # TODO: find from Peter how to combine multiple masks + batches

    attn_output, score = multihead_qkv_attention(score_returning, self.num_heads, query_states, key_states, value_states, masks, self.dropout)
    attn_output = self.out_proj(attn_output)

    if output_attentions
        return attn_output, score
    end
    return attn_output, nothing
end


# CLIPEncoderLayer
struct HGFCLIPEncoderLayer <: THModule
    mlp::HGFCLIPMLP
    layer_norm1::FakeTHLayerNorm
    layer_norm2::FakeTHLayerNorm
    self_attn::HGFCLIPAttention
end

@functor HGFCLIPEncoderLayer

"""
    HGFCLIPEncoderLayer(config::HGFCLIPTextConfig)
Create HGFCLIPEncoderLayer from HGFCLIPTextConfig config.

```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> enc_layer = Transformers.HuggingFace.HGFCLIPEncoderLayer(clip_config.text_config)
```
"""
HGFCLIPEncoderLayer(cfg::AbstractHGFConfig, args...; kwargs...) = HGFCLIPEncoderLayer(typeof(cfg), cfg, args...; kwargs...)
function HGFCLIPEncoderLayer(T::Type{HGFCLIPTextConfig}, config::HGFCLIPTextConfig)
    layer_norm1 = FakeTHLayerNorm(T, config, config.hidden_size; eps=Float32(config.layer_norm_eps))
    layer_norm2 = FakeTHLayerNorm(T, config, config.hidden_size; eps=Float32(config.layer_norm_eps))
    mlp = HGFCLIPMLP(T, config)
    self_attn = HGFCLIPAttention(T, config)
    HGFCLIPEncoderLayer(mlp, layer_norm1, layer_norm2, self_attn)
end

"""
    HGFCLIPEncoderLayer(hidden_states)
Evaluate the forward of HGFCLIPEncoderLayer given `hidden_states`. 

# Example for loading model from config
```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> enc_layer = Transformers.HuggingFace.HGFCLIPEncoderLayer(clip_config.text_config)
julia> state_dict = Transformers.load_state("openai/clip-vit-large-patch14")
julia> Transformers.HuggingFace.load_state!(enc_layer, state_dict.text_model.encoder.layers[1])
julia> enc_layer(zero(Array{Float32, 3}(undef, 768, 1, 1)))
```

"""
function (self::HGFCLIPEncoderLayer)(
    hidden_states::AbstractArray{T};                                 # embed_dim x seq_len x batch
    attention_mask::Union{Nothing}=nothing,                          # BatchedMask(LengthMask())
    causal_attention_mask::Union{CausalMask,Nothing}=nothing,                   # CausalMask()
    output_attentions::Bool=false
) where T

    residual = hidden_states
    hidden_states = self.layer_norm1(hidden_states)
    
    hidden_states, attn_weights = self.self_attn(
        hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
    )
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    
    if output_attentions
        return hidden_states, attn_weights
    end
    return hidden_states, nothing
end


# CLIPEncoder
struct HGFCLIPEncoder{T<:FakeTHModuleList} <: THModule
    config::HGFCLIPTextConfig
    layers::T
end

Functors.functor(::Type{<:HGFCLIPEncoder}, enc) = (layers=enc.layers, ), y->HGFCLIPEncoder(enc.config, y...)

"""
    HGFCLIPEncoder(config::HGFCLIPTextConfig)
Create HGFCLIPEncoder from a HGFCLIPTextConfig config.

```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> encoder = Transformers.HuggingFace.HGFCLIPEncoder(clip_config.text_config)
julia> state = Transformers.load_state("openai/clip-vit-large-patch14")
julia> Transformers.HuggingFace.load_state!(encoder, state.text_model.encoder)
```
"""
HGFCLIPEncoder(cfg::AbstractHGFConfig, args...; kwargs...) = HGFCLIPEncoder(typeof(cfg), cfg, args...; kwargs...)
function HGFCLIPEncoder(T::Type{HGFCLIPTextConfig}, config::HGFCLIPTextConfig)
    layers = FakeTHModuleList(
        [HGFCLIPEncoderLayer(T, config) for _ in 1:config.num_hidden_layers]
    )
    HGFCLIPEncoder(config, layers)
end

"""
    HGFCLIPEncoder(inputs_embeds)
Evaluate the forward of HGFCLIPEncoderLayer given `hidden_states`.

# Example for loading model from config
```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> encoder = Transformers.HuggingFace.HGFCLIPEncoder(clip_config.text_config)
julia> state_dict = Transformers.load_state("openai/clip-vit-large-patch14")
julia> Transformers.HuggingFace.load_state!(encoder, state_dict.text_model.encoder)
julia> encoder(randn(768, 1, 1)))
```

"""
function (self::HGFCLIPEncoder)(
    inputs_embeds;                                                    # hidden_size, sequence_length, batch_size
    attention_mask::Union{AbstractArray,Nothing}=nothing,
    causal_attention_mask::Union{CausalMask,Nothing}=nothing,
    output_attentions::Union{Bool,Nothing}=nothing,
    output_hidden_states::Union{Bool,Nothing}=nothing
)
    output_attentions = !isnothing(output_attentions) ? output_attentions : self.config.output_attentions
    output_hidden_states = !isnothing(output_hidden_states) ? output_hidden_states : self.config.output_hidden_states
    encoder_states = output_hidden_states ? () : nothing
    all_attentions = output_attentions ? () : nothing

    hidden_states = inputs_embeds
    for (idx, encoder_layer) in enumerate(self.layers)
        if output_hidden_states
            encoder_states = (encoder_states..., hidden_states)
        end
        # TODO: Gradient checkpointing & training code
        layer_outputs = encoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = layer_outputs[1]
        if output_attentions
            all_attentions = (all_attentions..., layer_outputs[2])
        end
    end

    if output_hidden_states
        encoder_states = (encoder_states..., hidden_states)
    end
    return (; last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)
end


# CLIPTextTransformer 
struct HGFCLIPTextTransformer <: THModule
    config::HGFCLIPTextConfig
    embeddings::HGFCLIPTextEmbeddings
    encoder::HGFCLIPEncoder
    final_layer_norm::FakeTHLayerNorm
end

Functors.functor(::Type{<:HGFCLIPTextTransformer}, m) = (embeddings=m.embeddings, encoder=m.encoder, final_layer_norm=m.final_layer_norm), y->HGFCLIPTextTransformer(m.config, y...)

"""
    HGFCLIPTextTransformer(config::HGFCLIPTextConfig)
Create HGFCLIPTextTransformer from HGFCLIPTextConfig config.

```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> transformer = Transformers.HuggingFace.HGFCLIPTextTransformer(clip_config.text_config)
julia> state = Transformers.load_state("openai/clip-vit-large-patch14")
julia> Transformers.HuggingFace.load_state!(transformer, state.text_model)
```
"""
HGFCLIPTextTransformer(cfg::AbstractHGFConfig, args...; kwargs...) = HGFCLIPTextTransformer(typeof(cfg), cfg, args...; kwargs...)
function HGFCLIPTextTransformer(T::Type{HGFCLIPTextConfig}, config::HGFCLIPTextConfig)
    embeddings = HGFCLIPTextEmbeddings(config)
    encoder = HGFCLIPEncoder(config)
    final_layer_norm = FakeTHLayerNorm(T, config, config.hidden_size; eps=Float32(config.layer_norm_eps))
    HGFCLIPTextTransformer(config, embeddings, encoder, final_layer_norm)
end

"""
    HGFCLIPTextTransformer(input_ids)
This function defines the forward pass of the HGFCLIPTextTransformer model.

```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> transformer = Transformers.HuggingFace.HGFCLIPTextTransformer(clip_config.text_config)
julia> transformer(cumsum(ones(Int32, 77)))
```
"""
function (self::HGFCLIPTextTransformer)(
    input_ids::AbstractArray;
    attention_mask::Union{Nothing}=nothing,
    position_ids::Union{AbstractArray,Nothing}=nothing,
    output_attentions::Union{Bool,Nothing}=nothing,
    output_hidden_states::Union{Bool,Nothing}=nothing
)
    output_attentions = !isnothing(output_attentions) ? output_attentions : self.config.output_attentions
    output_hidden_states = !isnothing(output_hidden_states) ? output_hidden_states : self.config.output_hidden_states

    hidden_states = self.embeddings(input_ids, position_ids)
    causal_attention_mask = NeuralAttentionlib.CausalMask()

    encoder_outputs = self.encoder(
        hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states
    )
    last_hidden_state = encoder_outputs[1]
    last_hidden_state = self.final_layer_norm(last_hidden_state)

    return (;last_hidden_state=last_hidden_state, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions,)
end


# CLIPModel 
abstract type HGFCLIPPreTrainedModel <: HGFPreTrainedModel end
struct HGFCLIPTextModel{E<:HGFCLIPTextTransformer} <: HGFCLIPPreTrainedModel
    text_model::E
end

@functor HGFCLIPTextModel

"""
    HGFCLIPTextModel(config::HGFCLIPTextConfig)
Create HGFCLIPTextModel from HGFCLIPTextConfig config.

```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> clip_text_model = Transformers.HuggingFace.HGFCLIPTextModel(clip_config.text_config)
julia> state = Transformers.load_state("openai/clip-vit-large-patch14")
julia> Transformers.HuggingFace.load_state!(clip_text_model, state)
```
"""
HGFCLIPTextModel(cfg::AbstractHGFConfig, args...; kwargs...) = HGFCLIPTextModel(typeof(cfg), cfg, args...; kwargs...)
function HGFCLIPTextModel(T::Type{HGFCLIPTextConfig}, config::HGFCLIPTextConfig)
    text_model = HGFCLIPTextTransformer(config)
    HGFCLIPTextModel(text_model)
end

"""
    HGFCLIPTextModel(input_ids)
This function defines the forward pass of the HGFCLIPTextModel model.

```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> clip_text_model = Transformers.HuggingFace.HGFCLIPTextModel(clip_config.text_config)
julia> clip_text_model(cumsum(ones(Int32, 77)))
```
"""
function (self::HGFCLIPTextModel)(
    input_ids::AbstractArray;
    attention_mask::Union{Nothing}=nothing,
    position_ids::Union{AbstractArray,Nothing}=nothing,
    output_attentions::Union{Bool,Nothing}=nothing,
    output_hidden_states::Union{Bool,Nothing}=nothing
)
    return self.text_model(input_ids; 
                           attention_mask=attention_mask, 
                           position_ids=position_ids,
                           output_attentions=output_attentions, 
                           output_hidden_states=output_hidden_states)
end

get_model_type(::Val{:clip}, ::Val{:model}) = HGFCLIPModel
