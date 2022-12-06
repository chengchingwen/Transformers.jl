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
julia> embeddings = Transformers.HuggingFace.HGFCLIPTextEmbeddings(clip_config.text_config)
julia> embeddings(input_ids)
```
"""
(self::HGFCLIPTextEmbeddings)(input_ids; position_ids = nothing) = self(input_ids, position_ids)
