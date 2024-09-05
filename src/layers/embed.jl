import Flux
using Functors
using NNlib
using ChainRulesCore

abstract type AbstractEmbedding end

"""
    Embed(hidden_size::Int, vocab_size::Int; scale = nothing)

An Embedding layer that take an array of integer / one-hot encoding and return a multi-dimensional array as
 embedded vectors and scale with `scale`.

See also: [`EmbedDecoder`](@ref)

# Example

```julia-repl
julia> embed = Embed(7, 10; scale = 100)
Embed(7, 10, scale = 100)

julia> embed([1,3,5])
7×3 Matrix{Float32}:
  0.86955    1.14728    0.43275
 -0.378461  -0.112709   3.33885
 -1.61534   -2.55506    1.08488
 -0.833164   0.565268  -1.32531
  0.820126  -5.11536   -0.75666
 -2.13458    1.25796   -1.47247
  3.20416    0.872459   0.980557

```
"""
struct Embed{F, E <: AbstractArray} <: AbstractEmbedding
    scale::F
    embeddings::E
end
@functor Embed (embeddings,)

Embed(embeddings::AbstractArray; scale = nothing) = Embed(scale, embeddings)
Embed(hidden_size::Int, vocab_size::Int; scale = nothing) = Embed(scale, randn(Float32, hidden_size, vocab_size))

(embed::Embed{Nothing})(x) = NNlib.gather(embed.embeddings, x)
function (embed::Embed)(x)
    y = NNlib.gather(embed.embeddings, x)
    return y .* convert(eltype(y), embed.scale)
end

function Base.show(io::IO, embed::Embed)
    print(io, "Embed(", size(embed.embeddings, 1), ", ", size(embed.embeddings, 2))
    !isnothing(embed.scale) && print(io, ", scale = ", embed.scale)
    print(io, ')')
end
@fluxlayershow Embed false

"""
    EmbedDecoder(embed::Embed; bias = false)

A layer that share weight with an embedding layer `embed` and return the logit.

See also: [`Embed`](@ref)
"""
struct EmbedDecoder{E<:AbstractEmbedding, B}
    embed::E
    bias::B
end
@functor EmbedDecoder

EmbedDecoder(embed::Embed; bias = false) = bias ?
    EmbedDecoder(embed, zeros(eltype(embed.embeddings), size(embed.embeddings, 1))) :
    EmbedDecoder(embed, nothing)

embed_decode(scale, embeddings, bias, x) = dense(nothing, embeddings', bias, x, scale)
embed_decode(scale::Nothing, embeddings, bias, x) = dense(nothing, embeddings', bias, x)

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(embed_decode), scale, embeddings, bias, x)
    _scale = isnothing(scale) ? true : scale
    y, dense_pullback = rrule(config, dense, nothing, embeddings', bias, x, _scale)
    function embed_decode_pullback(Ȳ)
        _, _, dembeddingsT, dbias, dx, _ = dense_pullback(Ȳ)
        if dembeddingsT isa ChainRulesCore.AbstractZero
            dembeddings = dembeddingsT
        else
            dembeddings = dembeddingsT'
        end
        return (NoTangent(), NoTangent(), dembeddings, dbias, dx)
    end
    return y, embed_decode_pullback
end

(e::EmbedDecoder{<:Embed})(x) = embed_decode(e.embed.scale, e.embed.embeddings, e.bias, x)

function Base.show(io::IO, e::EmbedDecoder)
    print(io, "EmbedDecoder(")
    show(io, e.embed)
    if !isnothing(e.bias)
        print(io, ", bias = true")
    end
    print(io, ')')
end
@fluxlayershow EmbedDecoder false

"""
    FixedLenPositionEmbed(hidden_size::Int, max_length::Int = 1024)

An trainable position embedding layer.

See also: [`SinCosPositionEmbed`](@ref)

# Example

```julia-repl
julia> pe = FixedLenPositionEmbed(7)
FixedLenPositionEmbed(7, 1024)

julia> pe(5)
7×5 Matrix{Float32}:
 -0.0330963    -0.0412815    -0.0110067    0.0299395   -0.0303213
  0.0203617    -0.000259752  -0.0300242    0.00573144   0.0147597
  0.00662918   -0.0222377    -9.40627f-5  -0.038285    -0.0467688
 -0.00358604    0.0344152     0.0101526   -0.00750311   0.0173139
  0.000689436   0.0116299    -0.00478128  -0.0331492    0.0148091
  0.000711651  -0.0198647    -0.0037188    0.00427536  -0.0172123
 -0.00987371   -0.0385056    -0.00103168   0.0578125    0.00286929

julia> pe([1,3])
7×2 Matrix{Float32}:
 -0.0330963    -0.0110067
  0.0203617    -0.0300242
  0.00662918   -9.40627f-5
 -0.00358604    0.0101526
  0.000689436  -0.00478128
  0.000711651  -0.0037188
 -0.00987371   -0.00103168

julia> pe(randn(3,3))
7×3 Matrix{Float32}:
 -0.0330963    -0.0412815    -0.0110067
  0.0203617    -0.000259752  -0.0300242
  0.00662918   -0.0222377    -9.40627f-5
 -0.00358604    0.0344152     0.0101526
  0.000689436   0.0116299    -0.00478128
  0.000711651  -0.0198647    -0.0037188
 -0.00987371   -0.0385056    -0.00103168

```
"""
struct FixedLenPositionEmbed{E <: AbstractArray} <: AbstractEmbedding
    embeddings::E
end
@functor FixedLenPositionEmbed

FixedLenPositionEmbed(hidden_size::Int, max_length::Int = 1024) =
    FixedLenPositionEmbed(init_weight(Float32, hidden_size, max_length))

(embed::FixedLenPositionEmbed)(x) = reshape(embed(size(x, 2)), Val(ndims(x)))
(embed::FixedLenPositionEmbed)(x::AbstractArray{<:Integer}) = NNlib.gather(embed.embeddings, x)
(embed::FixedLenPositionEmbed)(len::Int) = embed.embeddings[:, Base.OneTo(len)]

Base.show(io::IO, embed::FixedLenPositionEmbed) = (print(io, "FixedLenPositionEmbed"); print(io, size(embed.embeddings)))
@fluxlayershow FixedLenPositionEmbed false

"""
    SinCosPositionEmbed(hidden_size::Int)

The absolute sin cos postion embedding.

See also: [`FixedLenPositionEmbed`](@ref)

# Example

```julia-repl
julia> pe = SinCosPositionEmbed(7)
SinCosPositionEmbed(default_position_func(static(7)), 7, normalized = false)

julia> pe(5)
7×5 Matrix{Float32}:
 0.0  0.841471      0.909297      0.14112     -0.756802
 1.0  0.540302     -0.416147     -0.989992    -0.653644
 0.0  0.0719065     0.143441      0.214232     0.283915
 1.0  0.997411      0.989659      0.976783     0.95885
 0.0  0.00517945    0.0103588     0.0155378    0.0207164
 1.0  0.999987      0.999946      0.999879     0.999785
 0.0  0.000372759   0.000745519   0.00111828   0.00149104

julia> pe([1,3])
7×2 Matrix{Float32}:
 0.0   0.909297
 1.0  -0.416147
 0.0   0.143441
 1.0   0.989659
 0.0   0.0103588
 1.0   0.999946
 0.0   0.000745519

julia> pe(randn(3,3))
7×3 Matrix{Float64}:
 0.0  0.841471      0.909297
 1.0  0.540302     -0.416147
 0.0  0.0719065     0.143441
 1.0  0.997411      0.989659
 0.0  0.00517945    0.0103588
 1.0  0.999987      0.999946
 0.0  0.000372759   0.000745519

```
"""
struct SinCosPositionEmbed{F} <: AbstractEmbedding
    f::F
    hidden_size::Int
    normalized::Bool
end
SinCosPositionEmbed(hidden_size::Int, normalized::Bool = false) = SinCosPositionEmbed(
    NeuralAttentionlib.default_position_func(hidden_size), hidden_size, normalized)
SinCosPositionEmbed(f, hidden_size::Int) = SinCosPositionEmbed(f, hidden_size, false)

(embed::SinCosPositionEmbed)(x) = NeuralAttentionlib.get_sincos_position_embeddings(embed.f, embed.hidden_size, embed.normalized, x)

function Base.show(io::IO, embed::SinCosPositionEmbed)
    print(io, "SinCosPositionEmbed(")
    if embed.f isa Base.Fix1{typeof(NeuralAttentionlib.default_position_func)}
        print(io, "default_position_func(", embed.f.x, ')')
    else
        show(io, embed.f)
    end
    print(io, ", ", embed.hidden_size, ", normalized = ", embed.normalized, ')')
end
@fluxlayershow SinCosPositionEmbed false

struct RotaryPositionEmbed <: AbstractEmbedding end
(embed::RotaryPositionEmbed)(x) = NeuralAttentionlib.with_rotary_position_embedding(x)
@fluxlayershow RotaryPositionEmbed false

"""
    ApplyEmbed([apply = .+,] embed)

A layer that help to get embedding and apply on the input. Used with position embeddings.
"""
struct ApplyEmbed{F, E, I}
    apply::F
    embed::E
    indices::I
end
@functor ApplyEmbed (apply, embed)

ApplyEmbed(embed) = ApplyEmbed(.+, embed)
ApplyEmbed(apply, embed) = ApplyEmbed(apply, embed, identity)

function (e::ApplyEmbed)(x, indices = ChainRulesCore.ignore_derivatives(() -> e.indices(x)))
    embeddings = e.embed(indices)
    return e.apply(x, embeddings)
end

function Base.show(io::IO, e::ApplyEmbed)
    print(io, "ApplyEmbed(")
    if e.apply isa Broadcast.BroadcastFunction
        if Base.isoperator(nameof(e.apply.f))
            print(io, '.')
            show(io, e.apply.f)
        else
            show(io, e.apply.f)
            print(io, '.')
        end
    else
        show(io, e.apply)
    end
    print(io, ", ")
    show(io, e.embed)
    if !(e.indices isa typeof(identity))
        print(io, ", ")
        show(io, e.indices)
    end
    print(io, ')')
end
@fluxlayershow ApplyEmbed false
