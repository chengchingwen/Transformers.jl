using ..Transformers


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


# embedding
struct HGFCLIPTextEmbeddings{P<:FakeTHEmbedding,T<:FakeTHEmbedding,K<:AbstractArray} <: THModule
    position_embedding::P
    token_embedding::T
    position_ids::K
end

@functor HGFCLIPTextEmbeddings

"""
    HGFCLIPTextEmbeddings(clip_config.text_config)
Create CLIP Embeddings from the CLIPTextConfig.

# Example
```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> embeddings = Transformers.HuggingFace.HGFCLIPTextEmbeddings(clip_config.text_config)
```
"""
HGFCLIPTextEmbeddings(cfg::AbstractHGFConfig, args...; kwargs...) = HGFCLIPTextEmbeddings(typeof(cfg), cfg, args...; kwargs...)
function HGFCLIPTextEmbeddings(T::Type{HGFCLIPTextConfig}, config) 
    posi_emb = FakeTHEmbedding(T, config, config.max_position_embeddings, config.hidden_size)
    toke_emb = FakeTHEmbedding(T, config, config.vocab_size, config.hidden_size)
    pos_ids  = Array{Int64, 2}(undef, 1, config.max_position_embeddings)
    HGFCLIPTextEmbeddings(posi_emb, toke_emb, pos_ids)
end


@inline get_word_emb(emb::HGFCLIPTextEmbeddings, input_ids::AbstractArray{<:Integer}) = emb.token_embedding(input_ids)
# @inline get_word_emb(emb::HGFCLIPTextEmbeddings, input_embed::AbstractArray{T}) where T = input_embed # TODO: what does this do?

function get_position_emb(emb::HGFCLIPTextEmbeddings, ::Nothing)
    batch_size, token_size = size(emb.position_ids)
    pos_ids = reshape(emb.position_ids, (token_size, batch_size)) # put batch as last dim
    get_position_emb(emb, pos_ids .+ 1) # add 1 to pos_ids so it doesnot start from 0
end
@inline get_position_emb(emb::HGFCLIPTextEmbeddings, position_ids) = emb.position_embedding(position_ids)


# Forward definitions of HGFCLIPTextEmbeddings
"""
Call a CLIP embedding given input and position embeddings as inputs.

# Example
```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> embeddings = Transformers.HuggingFace.HGFCLIPTextEmbeddings(clip_config.text_config)
julia> embeddings(input_embeds, position_embeds)
```
"""
function (self::HGFCLIPTextEmbeddings)(inputs_embeds::AbstractArray{T}, position_embeds::AbstractArray{T}) where T
  return (inputs_embeds .+ position_embeds)
end

"""
    HGFCLIPTextEmbeddings()
Call a CLIP embeddings given input_ids and position_ids

# Example
```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> embeddings = Transformers.HuggingFace.HGFCLIPTextEmbeddings(clip_config.text_config)
julia> embeddings(input_ids, position_ids)
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
    HGFCLIPTextEmbeddings()
Call a CLIP embeddings given embeddings

# Example
```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> embeddings = Transformers.HuggingFace.HGFCLIPTextEmbeddings(clip_config.text_config)
julia> embeddings(input_ids)
```
"""
(self::HGFCLIPTextEmbeddings)(input_ids; position_ids = nothing) = self(input_ids, position_ids)


# HGFCLIPMLP
struct HGFCLIPMLP{F1<:FakeTHLinear, F2<:FakeTHLinear, A<:Any} <: THModule
    fc1::F1                                                      # nn.Linear
    fc2::F2                                                      # nn.Linear
    activation_fn::A                                             # QuickGELUActivation
end

"""
    HGFCLIPMLP()
Create CLIPMLP model from CLIPTextConfig config.

# Example for loading model from config
```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> clip_mlp = Transformers.HuggingFace.HGFCLIPMLP(clip_config.text_config)
```

# Example for loading CLIPMLP from a model in huggingface
```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> clip_mlp = Transformers.HuggingFace.HGFCLIPMLP(clip_config.text_config)
julia> state_dict = Transformers.load_state("openai/clip-vit-large-patch14")
julia> Transformers.HuggingFace.load_state!(clip_mlp, state.text_model.encoder.layers[1].mlp)
```
"""
HGFCLIPMLP(cfg::AbstractHGFConfig, args...; kwargs...) = HGFCLIPMLP(typeof(cfg), cfg, args...; kwargs...)
function HGFCLIPMLP(T::Type{HGFCLIPTextConfig}, config)
    fc1 = FakeTHLinear(T, config, config.hidden_size, config.intermediate_size)
    fc2 = FakeTHLinear(T, config, config.intermediate_size, config.hidden_size)
    act = ACT2FN[Symbol(config.hidden_act)]
    HGFCLIPMLP(fc1, fc2, act)
end

# Forward definitions of HGFCLIPTextEmbeddings
"""
    HGFCLIPMLP(x::AbstractArray)
CLIP MLP forward definition.

# Example of the forward call
```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> clip_mlp = Transformers.HuggingFace.HGFCLIPMLP(clip_config.text_config)
julia> out = clip_mlp(zero(Array{Float32, 3}(undef, 768, 1, 1)))
```
"""
# https://github.com/huggingface/transformers/blob/f68796bd603ef60173e093f50a1ecbb74bc2ba6b/src/transformers/models/clip/modeling_clip.py#L335
function (self::HGFCLIPMLP)(hidden_states::AbstractArray{T}) where T
    return self.fc2(self.activation_fn.(self.fc1(hidden_states)))
end

# Ask Peter: why do we need functors here? what the usecase?
# Functors.functor(::Type{<:HGFGPT2MLP}, mlp) = (c_fc = mlp.c_fc, c_proj = mlp.c_proj), y->HGFGPT2MLP(mlp.act, y...)


# HGFCLIPAttention
struct HGFCLIPAttention <: THModule
    k_proj::Any                                                  # nn.Linear; TODO type    
    v_proj::Any                                                  # nn.Linear; TODO type
    q_proj::Any                                                  # nn.Linear; TODO type
    out_proj::Any                                                # nn.Linear; TODO type
end

"""
    HGFCLIPAttention()
Create HGFCLIPAttention model from CLIPTextConfig config.

# Example for loading model from config
```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> clip_attn = HuggingFace.HGFCLIPAttention(clip_config.text_config)
```

# Example for loading CLIPAttention from a model in huggingface
```julia-repl
julia> clip_config = Transformers.load_config("openai/clip-vit-large-patch14")
julia> clip_attn = HuggingFace.HGFCLIPAttention(clip_config.text_config)
julia> state_dict = Transformers.load_state("openai/clip-vit-large-patch14")
julia> Transformers.HuggingFace.load_state!(clip_attn, state.text_model.encoder.layers[1].self_attn)
```
"""
HGFCLIPAttention(cfg::AbstractHGFConfig, args...; kwargs...) = HGFCLIPAttention(typeof(cfg), cfg, args...; kwargs...)
function HGFCLIPAttention(T::Type{HGFCLIPTextConfig}, config)
    # print(config)
    k_proj = FakeTHLinear(T, config, config.hidden_size, config.hidden_size)
    v_proj = FakeTHLinear(T, config, config.hidden_size, config.hidden_size)
    q_proj = FakeTHLinear(T, config, config.hidden_size, config.hidden_size)
    out_proj = FakeTHLinear(T, config, config.hidden_size, config.hidden_size)
    HGFCLIPAttention(k_proj, v_proj, q_proj, out_proj)
end

# https://github.com/huggingface/transformers/blob/f68796bd603ef60173e093f50a1ecbb74bc2ba6b/src/transformers/models/clip/modeling_clip.py#L231
function (self::HGFCLIPAttention)(
    hidden_states::AbstractArray{T};                             # embed_dim x seq_len x batch
    attention_mask::Union{Nothing, AbstractArray{T}}=nothing,
    causal_attention_mask::Union{Nothing, AbstractArray{T}}=nothing,
    output_attentions::Bool=false
) where T

    embed_dim, tgt_len, bsz  = size(hidden_states)

    # get query proj
    query_states = self.q_proj(hidden_states) * self.scale
end


# CLIPEncoderLayer 
struct HGFCLIPEncoderLayer <: THModule
    embed_dim::Any                                               # config.hidden_size; TODO type
    mlp::Any                                                     # CLIPMLP; TODO type
    layer_norm1::Any                                             # nn.LayerNorm; TODO type
    layer_norm2::Any                                             # nn.LayerNorm; TODO type
    self_attn::Any                                               # CLIPAttention; TODO type
end
function HGFCLIPEncoderLayer(config::Any)
    # print(config)
end


# CLIPEncoder 
struct HGFCLIPEncoder{T<:FakeTHModuleList} <: THModule
    layers::T                                                    # nn.ModuleList; TODO type
end
function HGFCLIPEncoder(config::Any)
    # print(config)
end


# CLIPTextModel 
struct HGFCLIPTextModel{P<:HGFCLIPTextEmbeddings} <: THModule
    embeddings::P                                                # CLIPTextEmbeddings
    encoder::Any                                                 # CLIPEncoder; TODO type 
    final_layer_norm::Any                                        # nn.LayerNorm; TODO type
end
function HGFCLIPTextModel(config::Any)
    embeddings = HGFCLIPTextEmbeddings(config)
    print(config)
end


# CLIPModel 
abstract type HGFCLIPPreTrainedModel <: HGFPreTrainedModel end
struct HGFCLIPModel{E<:HGFCLIPTextModel} <: HGFCLIPPreTrainedModel
    text_model::E
end
function HGFCLIPModel(config::Any)
    HGFCLIPTextModel(config.text_config)
end
get_model_type(::Val{:clip}, ::Val{:model}) = HGFCLIPModel
