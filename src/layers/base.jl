using Functors
using Static
using NeuralAttentionlib
using NeuralAttentionlib: $

function init_weight(::Type{T}, s...) where T
    weight = randn(T, s)
    weight .*= T(0.02)
    return weight
end

"""
    Fork(layers...)

A layer for applying each `layer`s to the same input and return a `Tuple`. For example `(Fork(dense1, dense2))(x)` is
 equivalent to `(dense1(x), dense2(x))`.
"""
struct Fork{T<:Tuple}
    layers::T
end
Fork(layers...) = Fork(layers)

@functor Fork

function (f::Fork)(x)
    return ntuple(i -> f.layers[i](x), Val(length(f.layers)))
end

function Base.show(io::IO, layer::Fork)
    print(io, "Fork")
    if layer.layers isa NTuple
        print(io, "<$(length(layer.layers))>(")
        show(io, layer.layers[1])
    else
        print(io, '(')
        show(io, layer.layers[1])
        for l in Base.tail(layer.layers)
            print(io, ", ")
            show(io, l)
        end
    end
    print(io, ')')
end
@fluxlayershow Fork false

"""
    NSplit(n::Integer, layer)

A layer for splitting the result of `layer` into `n` parts in the first dimension and return a `Tuple`. For
 example `(NSplit(2, dense))(x)` is equivalent to
 `y = dense(x); s1 = size(y, 1); (y[begin:div(s1, 2)-1, :], y[div(s1, 2):end, :]`.
"""
struct NSplit{N, L}
    n::N
    layer::L
end
NSplit(n::Integer, layer) = NSplit(static(n), layer)

@functor NSplit (layer,)

function nsplit(x, hdim, i)
    b = hdim * i
    a = b - hdim + 1
    cs = ntuple(i->Colon(), static(ndims(x)) - static(1))
    return @view x[a:b, cs...]
end

function (ns::NSplit)(x)
    y = ns.layer(x)
    ndim = ndims(y)
    hdim, r = divrem(size(y, 1), ns.n)
    @assert iszero(r) "NSplit try to split $(size(y,1)) in to $(Int(ns.n)) tensors"
    return ntuple(nsplit $ y $ hdim, ns.n)
end

function Base.show(io::IO, layer::NSplit)
    print(io, "NSplit<$(dynamic(layer.n))>(")
    show(io, layer.layer)
    print(io, ')')
end
@fluxlayershow NSplit false

######################################

bias_and_act(act, b, x) = act.(x .+ b)
bias_and_act(::Nothing, b, x) = x .+ b
bias_and_act(act, ::Nothing, x) = act.(x)
bias_and_act(::Nothing, ::Nothing, x) = x

dense(act, W, b, x) = bias_and_act(act, b, NeuralAttentionlib.scaled_matmul(W, x))

struct Dense{F, T, B}
    σ::F
    W::T
    b::B
end
@functor Dense (W, b)

Dense(w::AbstractArray) = Dense(nothing, w, nothing)
Dense(act, w::AbstractArray) = Dense(act, w, nothing)
Dense(w::AbstractArray, b::AbstractArray) = Dense(nothing, w, b)

Dense(din::Int, dout::Int; bias = true) = Dense(nothing, din, dout; bias)
function Dense(act, din::Int, dout::Int; bias = true)
    b = bias ? init_weight(Float32, dout) : nothing
    w = init_weight(Float32, dout, din)
    return Dense(act, w, b)
end

(d::Dense)(x) = dense(d.σ, d.W, d.b, x)

function Base.show(io::IO, d::Dense)
    print(io, "Dense(")
    if !isnothing(d.σ)
        print(io, "σ = ")
        show(io, d.σ)
        print(io, ", ")
    end
    print(io, "W = ")
    print(io, reverse(size(d.W)))
    print(io, ", b = ")
    print(io, !isnothing(d.b))
    print(io, ')')
end
@fluxlayershow Dense false

struct LayerNorm{A, B, F}
    α::A
    β::B
    ϵ::F
end
@functor LayerNorm (α, β)

(ln::LayerNorm)(x) = NeuralAttentionlib.layer_norm(ln.ϵ, ln.α, ln.β, x)

LayerNorm(hidden_size::Int; ϵ = 1e-7) = LayerNorm(ones(Float32, hidden_size), zeros(Float32, hidden_size), Float32(ϵ))

Base.show(io::IO, ln::LayerNorm) = print(io, "LayerNorm(", length(ln.α), ", ϵ = ", ln.ϵ, ')')
@fluxlayershow LayerNorm false

struct RMSLayerNorm{A, F}
    α::A
    ϵ::F
end
@functor RMSLayerNorm (α,)

(ln::RMSLayerNorm)(x) = NeuralAttentionlib.rms_layer_norm(ln.ϵ, ln.α, x)

RMSLayerNorm(hidden_size::Int; ϵ = 1e-7) = RMSLayerNorm(ones(Float32, hidden_size), Float32(ϵ))

Base.show(io::IO, ln::RMSLayerNorm) = print(io, "RMSLayerNorm(", length(ln.α), ", ϵ = ", ln.ϵ, ')')
@fluxlayershow RMSLayerNorm false
