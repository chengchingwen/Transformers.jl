using ..Transformers: lasttoken, firsttoken
using ..Layers
using ChainRulesCore
using Flux
using Functors
using NNlib
using NeuralAttentionlib
using NeuralAttentionlib: Matmul, NoMask

function pixel_embedding(x::AbstractArray{T, 4}, conv, class_emb::AbstractVector) where T
    patches = conv(x)
    W, H, C, N = size(patches)
    embs = similar(patches, C, W * H + 1, N)
    @view(embs[:, begin+1:end, :]) .= batched_transpose(reshape(patches, W * H, C, N))
    @view(embs[:, begin, :]) .= class_emb
    return embs
end
function ChainRulesCore.rrule(config::RuleConfig, ::typeof(pixel_embedding), x::AbstractArray{T, 4}, conv, class_emb::AbstractVector) where T
    conv_tape = rrule(config, conv, x)
    isnothing(conv_tape) && (conv_tape = rrule_via_ad(config, conv, x))
    patches, conv_pullback = conv_tape
    W, H, C, N = size(patches)
    embs = similar(patches, C, W * H + 1, N)
    @view(embs[:, begin+1:end, :]) .= batched_transpose(reshape(patches, W * H, C, N))
    @view(embs[:, begin, :]) .= class_emb
    function pixel_embedding_pullback(Ybar)
        Ȳ = unthunk(Ybar)
        ∂class_emb = reshape(sum(@view(Ȳ[:, begin, :]), dims=3), :)
        ∂patches = similar(Ȳ, W, H, C, N)
        ∂patches .= reshape(batched_transpose(@view Ȳ[:, begin+1:end, :]), W, H, C, N)
        ∂conv, ∂x = conv_pullback(∂patches)
        return (NoTangent(), ∂x, ∂conv, ∂class_emb)
    end
    return embs, pixel_embedding_pullback
end

struct CLIPPixelEmbed{C, E<:AbstractVector}
    conv::C
    class_emb::E
end
@functor CLIPPixelEmbed
@fluxlayershow CLIPPixelEmbed false

function Base.show(io::IO, e::CLIPPixelEmbed)
    print(io, "CLIPPixelEmbed(")
    show(io, e.conv)
    print(io, ", ")
    print(io, length(e.class_emb))
    print(io, ')')
end

(e::CLIPPixelEmbed)(x) = pixel_embedding(x, e.conv, e.class_emb)

struct CLIPTextPooler{D}
    dense::D
end
@functor CLIPTextPooler
@fluxlayershow CLIPTextPooler

(m::CLIPTextPooler)(x, seqmask = NoMask{NeuralAttentionlib.SEQUENCE}()) = m.dense(lasttoken(x, seqmask))
function (m::CLIPTextPooler)(nt::NamedTuple)
    seqmask = get(nt, :sequence_mask, NoMask{NeuralAttentionlib.SEQUENCE}())
    return merge(nt, (pooled = m(nt.hidden_state, seqmask),))
end

struct CLIPVisionPooler{N, D}
    norm::N
    dense::D
end
@functor CLIPVisionPooler
@fluxlayershow CLIPVisionPooler

(m::CLIPVisionPooler)(x) = m.dense(m.norm(firsttoken(x)))
(m::CLIPVisionPooler)(nt::NamedTuple) = merge(nt, (pooled = m(nt.hidden_state),))

function clip_cosine_similarity(text_embed, vision_embed, logit_scale)
    text_logit = (vision_embed' * text_embed) .* exp.(logit_scale)
    vision_logit = text_logit'
    return (text = text_logit, vision = vision_logit)
end
