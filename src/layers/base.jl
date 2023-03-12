using NNlib
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

bias_and_act!(act, b, y, x) = y .= act.(x .+ b)
bias_and_act!(::Nothing, b, y, x) = y .= x .+ b
bias_and_act!(act, ::Nothing, y, x) = y .= act.(x)
function bias_and_act!(::Nothing, ::Nothing, y, x)
    if y !== x
        y .= x
    end
    return y
end

bias_and_act(act, b, x) = act.(x .+ b)
bias_and_act(::Nothing, b, x) = x .+ b
bias_and_act(act, ::Nothing, x) = act.(x)
bias_and_act(::Nothing, ::Nothing, x) = x

function dense(act, W, b, x, s = true)
    y = NeuralAttentionlib.scaled_matmul(W, x, s)
    return bias_and_act!(act, b, y, y)
end

function gelu_forward_backward(x)
    α = NNlib.oftf(x, 0.044715)
    α2 = NNlib.oftf(x, 0.08943)
    λλ = NNlib.oftf(x, NNlib.gelu_2λ)
    x2 = x * x
    t = muladd(x2, α, one(x))
    Ω = NNlib.sigmoid_fast(λλ * x * t)
    dσ = conj(Ω * (1 - Ω))
    forward = x * Ω
    backward = muladd(dσ * λλ * muladd(x2, α2, t), x, Ω)
    return (forward, backward)
end

function swish_forward_backward(x)
    t = sigmoid_fast(x)
    Ω = x * t
    backward = muladd(t, (1 - Ω), Ω)
    return (Ω, backward)
end

_deriv_σ(Ω) = conj(Ω * (1 - Ω))
_deriv_relu(Ω) = Ω > 0
_deriv_tanh(Ω) = conj(1 - Ω^2)
act_pullback(act) = nothing
act_pullback(::typeof(gelu)) = gelu_forward_backward
act_pullback(::typeof(swish)) = swish_forward_backward
act_pullback(::typeof(relu)) = _deriv_relu
act_pullback(::typeof(elu)) = NNlib.deriv_elu
act_pullback(::typeof(tanh)) = _deriv_tanh
act_pullback(::typeof(NNlib.tanh_fast)) = _deriv_tanh
act_pullback(::typeof(σ)) = _deriv_σ
act_pullback(::typeof(NNlib.sigmoid_fast)) = _deriv_σ

require_x(pb) = false
require_x(::typeof(gelu_forward_backward)) = true
require_x(::typeof(swish_forward_backward)) = true

function _run_fw_bw!(act_fw_bw, x, dx, cidx)
    fw, bw = act_fw_bw(x)
    @inbounds dx[cidx] = bw
    return fw
end

_bias_rdims(dS, ::Nothing) = ()
function _bias_rdims(dS, b)
    N = ndims(dS)
    s = size(b)
    # https://github.com/JuliaDiff/ChainRules.jl/blob/158ca756ef99ccf3f1dde2e66b5855e8e68e0363/src/rulesets/Base/broadcast.jl#L326
    dims = ntuple(d -> get(s, d, 1) == 1 ? d : N+1, N)  # hack to get type-stable `dims`
    return dims
end

struct DensePullback{A, D, BK, MB, Y, B}
    ∇act::A
    dS::D
    back::BK
    mm_back::MB
    y::Y
    b::B
end
function (pb::DensePullback{Nothing, Nothing})(Ybar)
    Ȳ = unthunk(Ybar)
    _, _, db, dS = pb.back(Ȳ)
    _, dW, dx, _ = pb.mm_back(dS)
    return (NoTangent(), NoTangent(), dW, db, dx, NoTangent())
end
function (pb::DensePullback{A, D})(Ybar) where {A, D}
    dS, b = pb.dS, pb.b
    Ȳ = unthunk(Ybar)
    pb.dS .*= Ȳ
    db = isnothing(b) ? NoTangent() : sum(dS; dims = _bias_rdims(dS, b))
    _, dW, dx, _ = pb.mm_back(dS)
    return (NoTangent(), NoTangent(), dW, db, dx, NoTangent())
end
function (pb::DensePullback{A, Nothing})(Ybar) where A
    b = pb.b
    Ȳ = unthunk(Ybar)
    dS = pb.∇act.(pb.y) .* Ȳ
    db = isnothing(b) ? NoTangent() : sum(dS; dims = _bias_rdims(dS, b))
    _, dW, dx, _ = pb.mm_back(dS)
    return (NoTangent(), NoTangent(), dW, db, dx, NoTangent())
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(dense), act, W, b, x)
    y, dense_pullback = rrule(config, dense, act, W, b, x, true)
    pullback(Ȳ) = Base.front(dense_pullback(Ȳ))
    return y, pullback
end
function ChainRulesCore.rrule(config::RuleConfig, ::typeof(dense), act, W, b, x, s)
    S, mm_pullback = rrule(config, NeuralAttentionlib.scaled_matmul, W, x, s)
    ∇act = act_pullback(act)
    if isnothing(∇act)
        broadcast_tape = rrule(config, bias_and_act, act, b, S)
        isnothing(broadcast_tape) && (broadcast_tape = rrule_via_ad(config, bias_and_act, act, b, S))
        y, broadcast_pullback = broadcast_tape
        return y, DensePullback(nothing, nothing, broadcast_pullback, mm_pullback, y, nothing)
    elseif require_x(∇act)
        dS = similar(S)
        Sb = isnothing(b) ? S : Broadcast.broadcasted(+, S, b)
        S .=  _run_fw_bw!.(∇act, Sb, Ref(dS), CartesianIndices(dS))
        return S, DensePullback(∇act, dS, nothing, mm_pullback, S, b)
    else
        y = bias_and_act!(act, b, S, S)
        return y, DensePullback(∇act, nothing, nothing, mm_pullback, y, b)
    end
end

######################################

struct Dense{F, T, B}
    σ::F
    W::T
    b::B
end
@functor Dense (W, b)

Dense(w::AbstractArray) = Dense(nothing, w, nothing)
Dense(act, w::AbstractArray) = Dense(act, w, nothing)
Dense(w::AbstractArray, b::AbstractArray) = Dense(nothing, w, b)
Dense(w::AbstractArray, ::Nothing) = Dense(nothing, w, nothing)

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
