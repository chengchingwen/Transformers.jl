using Statistics
using ChainRulesCore
using Static
using PrimitiveOneHot
using NeuralAttentionlib: AbstractSeqMask, GenericSeqMask, LengthMask, RevLengthMask

using NNlib
import Flux.Losses
import Flux.Losses: crossentropy, logitcrossentropy

using Base.Broadcast: broadcasted, instantiate

if VERSION < v"1.7"
    Base.has_fast_linear_indexing(::Ref) = true
end

Refm(m, dest) = Ref(Broadcast.preprocess(dest, m))

_tn(m, s) = ntuple(identity, static(ndims(m)) - s)
_tn1(m) = _tn(m, static(1))
_tn2(m) = _tn(m, static(2))

_qlogp(q, p, ϵ, m) = @fastmath m * - Losses.xlogy(q, max(p, ϵ))

_sdiv(a, b) = @fastmath a / oftype(a, b)

Losses.crossentropy(ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask; ϵ = Losses.epseltype(ŷ)) =
    Losses.crossentropy(mean, ŷ, y, m; ϵ)
function Losses.crossentropy(agg::Union{typeof(sum), typeof(mean)},
                             ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask;
                             ϵ = Losses.epseltype(ŷ))
    Losses._check_sizes(ŷ, y)
    M = Broadcast.preprocess(ŷ, m)
    losses = sum(instantiate(broadcasted(_qlogp, y, ŷ, ϵ, M)); dims = _tn1(m), init = zero(eltype(ŷ)))
    loss = sum(instantiate(broadcasted(_sdiv, reshape(losses, :), Masks.lengths(m))))
    if agg isa typeof(mean)
        loss /= oftype(loss, length(losses))
    end
    return loss
end

function ∇_qlogp(dy, q, p, ϵ, m)
    dresult = @fastmath q / max(p, ϵ)
    dqlogp = - ifelse(iszero(q), zero(dresult), dresult)
    return @fastmath m * (dy * dqlogp)
end

function ChainRulesCore.rrule(::typeof(crossentropy), agg::Union{typeof(sum), typeof(mean)},
                              ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask;
                              ϵ = Losses.epseltype(ŷ))
    Losses._check_sizes(ŷ, y)
    M = Broadcast.preprocess(ŷ, m)
    losses = sum(instantiate(broadcasted(_qlogp, y, ŷ, ϵ, M)); dims = _tn1(m), init = zero(eltype(ŷ)))
    ls = Masks.lengths(m)
    loss = sum(instantiate(broadcasted(_sdiv, reshape(losses, :), ls)))
    if agg isa typeof(mean)
        loss /= oftype(loss, length(losses))
    end
    scale = oftype(loss, agg isa typeof(mean) ? length(ls) : 1)
    function crossentropy_pullback(Ybar)
        Ȳ = unthunk(Ybar) / scale
        dlosses = reshape(_sdiv.(Ȳ, ls), (ntuple(one, static(ndims(m)) - static(1))..., length(ls)))
        dy = ∇_qlogp.(dlosses, y, ŷ, ϵ, m)
        return (NoTangent(), NoTangent(), dy, NoTangent(), NoTangent())
    end
    return loss, crossentropy_pullback
end

function _bcglog(a, c, cid, ϵ, m)
    I = CartesianIndex(c, cid)
    return @fastmath @inbounds m[I] * -log(max(a[I], ϵ))
end

function ∇_bcglog!(dy, dl, a, c, cid, ϵ, m)
    I = CartesianIndex(c, cid)
    dqlogp = @fastmath - @inbounds inv(max(a[I], ϵ))
    lid = @inbounds cid[length(cid)]
    @fastmath @inbounds dy[I] = m[I] * (dl[lid] * dqlogp)
end

Losses.crossentropy(ŷ::AbstractArray, y::OneHotArray, m::AbstractSeqMask; ϵ = Losses.epseltype(ŷ)) =
    Losses.crossentropy(mean, ŷ, y, m; ϵ)
function Losses.crossentropy(agg::Union{typeof(sum), typeof(mean)},
                             ŷ::AbstractArray, y::OneHotArray, m::AbstractSeqMask;
                             ϵ = Losses.epseltype(ŷ))
    Losses._check_sizes(ŷ, y)
    c = reinterpret(Int32, y)
    refm = Refm(m, ŷ)
    losses = sum(instantiate(broadcasted(_bcglog, Ref(ŷ), c, CartesianIndices(c), ϵ, refm));
                 dims = _tn2(m), init = zero(eltype(ŷ)))
    loss = sum(instantiate(broadcasted(_sdiv, reshape(losses, :), Masks.lengths(m))))
    if agg isa typeof(mean)
        loss /= oftype(loss, length(losses))
    end
    return loss
end

_z(a, b) = 0

function ChainRulesCore.rrule(::typeof(crossentropy), agg::Union{typeof(sum), typeof(mean)},
                              ŷ::AbstractArray, y::OneHotArray, m::AbstractSeqMask;
                              ϵ = Losses.epseltype(ŷ))
    Losses._check_sizes(ŷ, y)
    c = reinterpret(Int32, y)
    refm = Refm(m, ŷ)
    losses = sum(instantiate(broadcasted(_bcglog, Ref(ŷ), c, CartesianIndices(c), ϵ, refm));
                 dims = _tn2(m), init = zero(eltype(ŷ)))
    ls = Masks.lengths(m)
    loss = sum(instantiate(broadcasted(_sdiv, reshape(losses, :), ls)))
    if agg isa typeof(mean)
        loss /= oftype(loss, length(losses))
    end
    scale = oftype(loss, agg isa typeof(mean) ? length(ls) : 1)
    function crossentropy_pullback(Ybar)
        Ȳ = unthunk(Ybar) / scale
        dlosses = _sdiv.(Ȳ, ls)
        dy = fill!(similar(ŷ), 0)
        mapreduce(identity, _z, instantiate(broadcasted(
            ∇_bcglog!, Ref(dy), Ref(dlosses), Ref(ŷ), c, CartesianIndices(c), ϵ, refm)); init = 0)
        return (NoTangent(), NoTangent(), dy, NoTangent(), NoTangent())
    end
    return loss, crossentropy_pullback
end

_qp(q, logp, m) = @fastmath m * (- q * logp)

Losses.logitcrossentropy(ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask) =
    Losses.logitcrossentropy(mean, ŷ, y, m)
function Losses.logitcrossentropy(agg::Union{typeof(sum), typeof(mean)},
                                  ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask)
    Losses._check_sizes(ŷ, y)
    logp = logsoftmax(ŷ; dims = 1)
    M = Broadcast.preprocess(ŷ, m)
    losses = sum(instantiate(broadcasted(_qp, y, logp, M)); dims = _tn1(m), init = zero(eltype(ŷ)))
    loss = sum(instantiate(broadcasted(_sdiv, reshape(losses, :), Masks.lengths(m))))
    if agg isa typeof(mean)
        loss /= oftype(loss, length(losses))
    end
    return loss
end

function ChainRulesCore.rrule(::typeof(logitcrossentropy), agg::Union{typeof(sum), typeof(mean)},
                              ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask)
    Losses._check_sizes(ŷ, y)
    logp = logsoftmax(ŷ; dims = 1)
    M = Broadcast.preprocess(ŷ, m)
    losses = sum(instantiate(broadcasted(_qp, y, logp, M)); dims = _tn1(m), init = zero(eltype(ŷ)))
    ls = Masks.lengths(m)
    loss = sum(instantiate(broadcasted(_sdiv, reshape(losses, :), ls)))
    scale = oftype(loss, agg isa typeof(mean) ? length(ls) : 1)
    function logitcrossentropy_pullback(Ybar)
        Ȳ = unthunk(Ybar) / scale
        dlosses = reshape(_sdiv.(Ȳ, ls), (ntuple(one, static(ndims(m)) - static(1))..., length(ls)))
        dlogp = _qp.(y, dlosses, m)
        dy = NNlib.∇logsoftmax_data(dlogp, logp; dims = 1)
        return (NoTangent(), NoTangent(), dy, NoTangent(), NoTangent())
    end
    return loss, logitcrossentropy_pullback
end

function _bcg(a, c, cid, m)
    I = CartesianIndex(c, cid)
    return @fastmath @inbounds m[I] * - a[I]
end

function ∇_bcg!(dy, dl, c, cid, m)
    I = CartesianIndex(c, cid)
    lid = @inbounds cid[length(cid)]
    @fastmath @inbounds dy[I] = m[I] * - dl[lid]
end

_exp(x) = Base.FastMath.exp_fast(x)
Base.has_fast_linear_indexing(::Broadcast.Broadcasted{<:Union{Nothing, Broadcast.BroadcastStyle}, A, typeof(_exp)}) where {A} = false

Losses.logitcrossentropy(ŷ::AbstractArray, y::OneHotArray, m::AbstractSeqMask) =
    Losses.logitcrossentropy(mean, ŷ, y, m)
function Losses.logitcrossentropy(agg::Union{typeof(sum), typeof(mean)},
                                  ŷ::AbstractArray, y::OneHotArray, m::AbstractSeqMask)
    Losses._check_sizes(ŷ, y)
    xmax = maximum(ŷ; dims = 1)
    xdiff = instantiate(broadcasted(Base.FastMath.sub_fast, ŷ, xmax))
    sexp = sum(instantiate(broadcasted(_exp, xdiff)); dims = 1, init = zero(eltype(ŷ)))
    logp = instantiate(broadcasted(Base.FastMath.sub_fast, xdiff, broadcasted(Base.FastMath.log_fast, sexp)))
    c = reinterpret(Int32, y)
    refm = Refm(m, ŷ)
    losses = sum(instantiate(broadcasted(_bcg, Ref(logp), c, CartesianIndices(c), refm));
                 dims = _tn2(m), init = zero(eltype(ŷ)))
    loss = sum(instantiate(broadcasted(_sdiv, reshape(losses, :), Masks.lengths(m))))
    if agg isa typeof(mean)
        loss /= oftype(loss, length(losses))
    end
    return loss
end

# https://github.com/FluxML/NNlib.jl/blob/0b64dc11e6ba47707c43cf668663e48a615c85bb/src/softmax.jl#L122
∇logsoftmax_data!(dy, y; dims = 1) = @fastmath dy .-= sum(dy; dims) .* exp.(y)

function ChainRulesCore.rrule(::typeof(logitcrossentropy), agg::Union{typeof(sum), typeof(mean)},
                              ŷ::AbstractArray, y::OneHotArray, m::AbstractSeqMask)
    Losses._check_sizes(ŷ, y)
    xmax = maximum(ŷ; dims = 1)
    xdiff = instantiate(broadcasted(Base.FastMath.sub_fast, ŷ, xmax))
    sexp = sum(instantiate(broadcasted(_exp, xdiff)); dims = 1, init = zero(eltype(ŷ)))
    logp = instantiate(broadcasted(Base.FastMath.sub_fast, xdiff, broadcasted(Base.FastMath.log_fast, sexp)))
    c = reinterpret(Int32, y)
    refm = Refm(m, ŷ)
    losses = sum(instantiate(broadcasted(_bcg, Ref(logp), c, CartesianIndices(c), refm));
                 dims = _tn2(m), init = zero(eltype(ŷ)))
    ls = Masks.lengths(m)
    loss = sum(instantiate(broadcasted(_sdiv, reshape(losses, :), ls)))
    scale = oftype(loss, agg isa typeof(mean) ? length(ls) : 1)
    function logitcrossentropy_pullback(Ybar)
        Ȳ = unthunk(Ybar) / scale
        dlosses = reshape(_sdiv.(Ȳ, ls), (ntuple(one, static(ndims(m)) - static(1))..., length(ls)))
        dlogp = fill!(similar(ŷ), 0)
        mapreduce(identity, _z, instantiate(broadcasted(
            ∇_bcg!, Ref(dlogp), Ref(dlosses), c, CartesianIndices(c), refm)); init = 0)
        dy = ∇logsoftmax_data!(dlogp, logp; dims = 1)
        return (NoTangent(), NoTangent(), dy, NoTangent(), NoTangent())
    end
    return loss, logitcrossentropy_pullback
end
