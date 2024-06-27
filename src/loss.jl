using Statistics
using ChainRulesCore
using Static
using PrimitiveOneHot
using PrimitiveOneHot: AbstractOneHotArray
using NeuralAttentionlib: AbstractSeqMask, GenericSeqMask, LengthMask, RevLengthMask, NoMask

using NNlib
import Flux.Losses
import Flux.Losses: crossentropy, logitcrossentropy

using Base.Broadcast: broadcasted, instantiate

if VERSION < v"1.7"
    Base.has_fast_linear_indexing(::Ref) = true
end

_tn(m, s) = ntuple(identity, static(ndims(m)) - s)
_tn1(m) = _tn(m, static(1))
_tn2(m) = _tn(m, static(2))

_qlogp(q, p, ϵ, m) = @fastmath m * - Losses.xlogy(q, max(p, ϵ))

_sdiv(a, b) = @fastmath a / oftype(a, b)

"""
    crossentropy(ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask; ϵ)
    crossentropy(sum, ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask; ϵ)

`Flux.crossentropy` with an extra sequence mask for masking out non-needed token loss. `y` is the labels. By default
 it take the mean by dividing the number of valid tokens. This can be change to simply sum the valid losses by add
 the first argument `sum`. See also [`safe_crossentropy`](@ref)
"""
Losses.crossentropy(ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask; ϵ = Losses.epseltype(ŷ)) =
    Losses.crossentropy(mean, ŷ, y, m; ϵ)
function Losses.crossentropy(agg::Union{typeof(sum), typeof(mean)},
                             ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask;
                             ϵ = Losses.epseltype(ŷ))
    Losses._check_sizes(ŷ, y)
    M = Masks.GetIndexer(m, size(ŷ))
    losses = sum(instantiate(broadcasted(_qlogp, y, ŷ, ϵ, M)); dims = _tn1(m), init = zero(eltype(ŷ)))
    loss = sum(instantiate(broadcasted(_sdiv, reshape(losses, :), Masks.lengths(m))))
    if agg isa typeof(mean)
        loss /= oftype(loss, length(losses))
    end
    return loss
end
function Losses.crossentropy(agg::Union{typeof(sum), typeof(mean)},
                             ŷ::AbstractArray, y::OneHotArray, m::AbstractSeqMask;
                             ϵ = Losses.epseltype(ŷ))
    Losses._check_sizes(ŷ, y)
    c = reinterpret(Int32, y)
    return _unsafe_crossentropy(agg, ŷ, c, m; ϵ)
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
    M = Masks.GetIndexer(m, size(ŷ))
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

function _unsafe_crossentropy(agg::Union{typeof(sum), typeof(mean)}, ŷ::AbstractArray, c::AbstractArray{<:Integer}, m::AbstractSeqMask; ϵ = Losses.epseltype(ŷ))
    refm = Ref(Masks.GetIndexer(m, size(ŷ)))
    losses = sum(instantiate(broadcasted(_bcglog, Ref(ŷ), c, CartesianIndices(c), ϵ, refm));
                 dims = _tn2(m), init = zero(eltype(ŷ)))
    loss = sum(instantiate(broadcasted(_sdiv, reshape(losses, :), Masks.lengths(m))))
    if agg isa typeof(mean)
        loss /= oftype(loss, length(losses))
    end
    return loss
end

function _check_sizes_int(ŷ::AbstractArray, c::AbstractArray{<:Integer})
    for d in 1:max(ndims(ŷ), ndims(c))
        size(ŷ, d + 1) == size(c, d) || throw(DimensionMismatch(
            "loss function with integer label expects Base.tail(size(ŷ)) = $(Base.tail(size(ŷ))) to match size(c) = $(size(c))"))
    end
end
ChainRulesCore.@non_differentiable _check_sizes_int(ŷ, c)

_z(a, b) = false

function ChainRulesCore.rrule(::typeof(_unsafe_crossentropy), agg::Union{typeof(sum), typeof(mean)},
                              ŷ::AbstractArray, c::AbstractArray{<:Integer}, m::AbstractSeqMask;
                              ϵ = Losses.epseltype(ŷ))
    refm = Ref(Masks.GetIndexer(m, size(ŷ)))
    losses = sum(instantiate(broadcasted(_bcglog, Ref(ŷ), c, CartesianIndices(c), ϵ, refm));
                 dims = _tn2(m), init = zero(eltype(ŷ)))
    ls = Masks.lengths(m)
    loss = sum(instantiate(broadcasted(_sdiv, reshape(losses, :), ls)))
    if agg isa typeof(mean)
        loss /= oftype(loss, length(losses))
    end
    scale = oftype(loss, agg isa typeof(mean) ? length(ls) : 1)
    function _unsafe_crossentropy_pullback(Ybar)
        Ȳ = unthunk(Ybar) / scale
        dlosses = _sdiv.(Ȳ, ls)
        dy = fill!(similar(ŷ), 0)
        mapreduce(identity, _z, instantiate(broadcasted(
            ∇_bcglog!, Ref(dy), Ref(dlosses), Ref(ŷ), c, CartesianIndices(c), ϵ, refm)); init = false)
        return (NoTangent(), NoTangent(), dy, NoTangent(), NoTangent())
    end
    return loss, _unsafe_crossentropy_pullback
end

"""
    unsafe_crossentropy(ŷ::AbstractArray, y::AbstractArray{<:Integer}, m::AbstractSeqMask; ϵ)
    unsafe_crossentropy(sum, ŷ::AbstractArray, y::AbstractArray{<:Integer}, m::AbstractSeqMask; ϵ)

Compute [`crossentropy`](@ref) with integer labels. The prefix "unsafe" means that if `y` contain any number larger
 than the first dimension of `ŷ`, the behavior is undefined. See also [`safe_crossentropy`](@ref).
"""
unsafe_crossentropy(ŷ::AbstractArray, c::AbstractArray{<:Integer}, m::AbstractSeqMask; ϵ = Losses.epseltype(ŷ)) =
    unsafe_crossentropy(mean, ŷ, c, m; ϵ)
function unsafe_crossentropy(agg::Union{typeof(sum), typeof(mean)},
                             ŷ::AbstractArray, c::AbstractArray{<:Integer}, m::AbstractSeqMask;
                             ϵ = Losses.epseltype(ŷ))
    _check_sizes_int(ŷ, c)
    return _unsafe_crossentropy(agg, ŷ, c, m; ϵ)
end

_qp(q, logp, m) = @fastmath m * (- q * logp)

"""
    logitcrossentropy(ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask)
    logitcrossentropy(sum, ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask)

`Flux.logitcrossentropy` with an extra sequence mask for masking out non-needed token loss. `y` is the labels. By
 default it take the mean by dividing the number of valid tokens. This can be change to simply sum the valid losses
 by add the first argument `sum`. See also [`safe_logitcrossentropy`](@ref)
"""
Losses.logitcrossentropy(ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask) =
    Losses.logitcrossentropy(mean, ŷ, y, m)
function Losses.logitcrossentropy(agg::Union{typeof(sum), typeof(mean)},
                                  ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask)
    Losses._check_sizes(ŷ, y)
    logp = logsoftmax(ŷ; dims = 1)
    M = Masks.GetIndexer(m, size(ŷ))
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
    M = Masks.GetIndexer(m, size(ŷ))
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

function _unsafe_logitcrossentropy(agg::Union{typeof(sum), typeof(mean)},
                                  ŷ::AbstractArray, c::AbstractArray{<:Integer}, m::AbstractSeqMask)
    xmax = maximum(ŷ; dims = 1)
    xdiff = instantiate(broadcasted(Base.FastMath.sub_fast, ŷ, xmax))
    sexp = sum(instantiate(broadcasted(_exp, xdiff)); dims = 1, init = zero(eltype(ŷ)))
    logp = instantiate(broadcasted(Base.FastMath.sub_fast, xdiff, broadcasted(Base.FastMath.log_fast, sexp)))
    refm = Ref(Masks.GetIndexer(m, size(ŷ)))
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

function ChainRulesCore.rrule(::typeof(_unsafe_logitcrossentropy), agg::Union{typeof(sum), typeof(mean)},
                              ŷ::AbstractArray, c::AbstractArray{<:Integer}, m::AbstractSeqMask)
    xmax = maximum(ŷ; dims = 1)
    xdiff = instantiate(broadcasted(Base.FastMath.sub_fast, ŷ, xmax))
    sexp = sum(instantiate(broadcasted(_exp, xdiff)); dims = 1, init = zero(eltype(ŷ)))
    logp = instantiate(broadcasted(Base.FastMath.sub_fast, xdiff, broadcasted(Base.FastMath.log_fast, sexp)))
    refm = Ref(Masks.GetIndexer(m, size(ŷ)))
    losses = sum(instantiate(broadcasted(_bcg, Ref(logp), c, CartesianIndices(c), refm));
                 dims = _tn2(m), init = zero(eltype(ŷ)))
    ls = Masks.lengths(m)
    loss = sum(instantiate(broadcasted(_sdiv, reshape(losses, :), ls)))
    scale = oftype(loss, agg isa typeof(mean) ? length(ls) : 1)
    function _unsafe_logitcrossentropy_pullback(Ybar)
        Ȳ = unthunk(Ybar) / scale
        dlosses = reshape(_sdiv.(Ȳ, ls), (ntuple(one, static(ndims(m)) - static(1))..., length(ls)))
        dlogp = fill!(similar(ŷ), 0)
        mapreduce(identity, _z, instantiate(broadcasted(
            ∇_bcg!, Ref(dlogp), Ref(dlosses), c, CartesianIndices(c), refm)); init = false)
        dy = ∇logsoftmax_data!(dlogp, logp; dims = 1)
        return (NoTangent(), NoTangent(), dy, NoTangent(), NoTangent())
    end
    return loss, _unsafe_logitcrossentropy_pullback
end

"""
    unsafe_logitcrossentropy(ŷ::AbstractArray, y::AbstractArray{<:Integer}, m::AbstractSeqMask)
    unsafe_logitcrossentropy(sum, ŷ::AbstractArray, y::AbstractArray{<:Integer}, m::AbstractSeqMask)

Compute [`logitcrossentropy`](@ref) with integer labels. The prefix "unsafe" means that if `y` contain any number
 larger than the first dimension of `ŷ`, the behavior is undefined. See also [`safe_logitcrossentropy`](@ref).
"""
unsafe_logitcrossentropy(ŷ::AbstractArray, c::AbstractArray{<:Integer}, m::AbstractSeqMask) =
    unsafe_logitcrossentropy(mean, ŷ, c, m)
function unsafe_logitcrossentropy(agg::Union{typeof(sum), typeof(mean)},
                                  ŷ::AbstractArray, c::AbstractArray{<:Integer}, m::AbstractSeqMask)
    _check_sizes_int(ŷ, c)
    return _unsafe_logitcrossentropy(agg, ŷ, c, m)
end

Losses.logitcrossentropy(ŷ::AbstractArray, y::OneHotArray, m::AbstractSeqMask) =
    Losses.logitcrossentropy(mean, ŷ, y, m)
function Losses.logitcrossentropy(agg::Union{typeof(sum), typeof(mean)},
                                  ŷ::AbstractArray, y::OneHotArray, m::AbstractSeqMask)
    Losses._check_sizes(ŷ, y)
    c = reinterpret(Int32, y)
    return _unsafe_logitcrossentropy(agg, ŷ, c, m)
end

"""
    safe_crossentropy(ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask; ϵ)
    safe_crossentropy(sum, ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask; ϵ)

[`crossentropy`](@ref). If the label `y` is an integer array, then it would also call `maximum` on the label to
 make sure no label number is large then the first dimension of `ŷ`. See also [`unsafe_crossentropy`](@ref).
"""
safe_crossentropy(ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask; ϵ = Losses.epseltype(ŷ)) =
    safe_crossentropy(mean, ŷ, y, m; ϵ)
function safe_crossentropy(
    agg::Union{typeof(sum), typeof(mean)}, ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask;
    ϵ = Losses.epseltype(ŷ))
    return crossentropy(agg, ŷ, y, m; ϵ)
end
function safe_crossentropy(
    agg::Union{typeof(sum), typeof(mean)}, ŷ::AbstractArray, c::AbstractArray{<:Integer}, m::AbstractSeqMask; ϵ = Losses.epseltype(ŷ))
    ChainRulesCore.ignore_derivatives() do
        k = maximum(c)
        n = size(ŷ, 1)
        n < k && throw(ArgumentError("logits have $n possible outputs while there are at least $k labels"))
        return nothing
    end
    return unsafe_crossentropy(agg, ŷ, c, m; ϵ)
end

"""
    safe_logitcrossentropy(ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask)
    safe_logitcrossentropy(sum, ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask)

[`logitcrossentropy`](@ref). If the label `y` is an integer array, then it would also call `maximum` on the label to
 make sure no label number is large then the first dimension of `ŷ`. See also [`unsafe_logitcrossentropy`](@ref).
"""
safe_logitcrossentropy(ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask) =
    safe_logitcrossentropy(mean, ŷ, y, m)
function safe_logitcrossentropy(
    agg::Union{typeof(sum), typeof(mean)}, ŷ::AbstractArray, y::AbstractArray, m::AbstractSeqMask)
    return logitcrossentropy(agg, ŷ, y, m)
end
function safe_logitcrossentropy(
    agg::Union{typeof(sum), typeof(mean)}, ŷ::AbstractArray, c::AbstractArray{<:Integer}, m::AbstractSeqMask)
    ChainRulesCore.ignore_derivatives() do
        k = maximum(c)
        n = size(ŷ, 1)
        n < k && throw(ArgumentError("logits have $n possible outputs while there are at least $k labels"))
        return nothing
    end
    return unsafe_logitcrossentropy(agg, ŷ, c, m)
end

isidarray(x::AbstractArray) = (eltype(x) <: Bool) ⊻ (eltype(x) <: Union{Integer, AbstractOneHotArray})
lengthdim(x::AbstractArray) = isidarray(x) ? 1 : 2
lengthdimlength(x::AbstractArray) = size(x, lengthdim(x))
lengthdimfirstindex(x::AbstractArray) = firstindex(x, lengthdim(x))
lengthdimlastindex(x::AbstractArray) = lastindex(x, lengthdim(x))

ChainRulesCore.@non_differentiable isidarray(x)
ChainRulesCore.@non_differentiable lengthdim(x)
ChainRulesCore.@non_differentiable lengthdimlength(x)
ChainRulesCore.@non_differentiable lengthdimfirstindex(x)
ChainRulesCore.@non_differentiable lengthdimlastindex(x)

function _findbound(f::Union{typeof(min), typeof(max)}, x, m::AbstractSeqMask)
    if isidarray(x)
        x = reshape(x, 1, size(x)...)
    end
    len_batch = Base.tail(size(x))
    len = first(len_batch)
    batch = Base.tail(len_batch)
    if isidarray(x)
        R = similar(x, Int32, 1, batch...)
        R′ = reshape(R, 1, size(R)...)
    else
        R = similar(x, Int32, 1, 1, batch...)
        R′ = R
    end
    init = Int32(typeof(f) <: typeof(min) ? len : 1)
    fill!(R, init)
    M = Masks.GetIndexer(m, (1, len_batch...))
    I = reshape(Base.OneTo{Int32}(len), (1, len, ntuple(one, Val(length(batch)))...))
    A = instantiate(broadcasted(ifelse, M, I, init))
    Base.mapreducedim!(identity, f, R′, A)
    return R
end
ChainRulesCore.@non_differentiable _findbound(f, x, m)

function _unsafe_tokenselect(x::OneHotArray, i)
    @assert isone(size(i, 1))
    return OneHotArray(_unsafe_tokenselect(parent(x), reshape(i, Base.tail(size(i)))))
end
function _unsafe_tokenselect(x::AbstractArray, i)
    if isidarray(x)
        len_batch = size(x)
    else
        @assert isone(size(i, 1))
        s = size(x)
        fdim = first(s)
        len_batch = Base.tail(s)
    end
    len = first(len_batch)
    batch = Base.tail(len_batch)
    bs = ntuple(Val(length(batch))) do bi
        b = batch[bi]
        reshape(Base.OneTo{Int32}(b), ntuple(one, Val(lengthdim(x)))..., ntuple(bj -> bi == bj ? b : 1, Val(length(batch)))...)
    end
    if isidarray(x)
        shape = size(i)
        indices = (i, bs...)
    else
        shape = (fdim, Base.tail(size(i))...)
        indices = (Base.OneTo{Int32}(fdim), i, bs...)
    end
    y = similar(x, shape)
    broadcast!(getindex, y, Ref(x), indices...)
    return y
end
ChainRulesCore.@non_differentiable _unsafe_tokenselect(x::OneHotArray, i)
function _unsafe_singletokenselect(x::AbstractArray, i)
    @assert isone(size(i, lengthdim(x)))
    y = _unsafe_tokenselect(x, i)
    if isidarray(x)
        shape = Base.tail(size(x))
    else
        shape = (size(y, 1), Base.tail(Base.tail(size(i)))...)
    end
    return reshape(y, shape)
end
function ∇_getindex!(dx, dy, i1, i, is...)
    @inbounds dx[i1, i, is...] = dy
end
function ChainRulesCore.rrule(::typeof(_unsafe_tokenselect), x::AbstractArray, i)
    if isidarray(x)
        len_batch = size(x)
    else
        @assert isone(size(i, 1))
        s = size(x)
        fdim = first(s)
        len_batch = Base.tail(s)
    end
    len = first(len_batch)
    batch = Base.tail(len_batch)
    bs = ntuple(Val(length(batch))) do bi
        b = batch[bi]
        reshape(Base.OneTo{Int32}(b), ntuple(one, Val(lengthdim(x)))..., ntuple(bj -> bi == bj ? b : 1, Val(length(batch)))...)
    end
    if isidarray(x)
        shape = size(i)
        indices = (i, bs...)
    else
        shape = (fdim, Base.tail(size(i))...)
        indices = (Base.OneTo{Int32}(fdim), i, bs...)
    end
    y = similar(x, shape)
    broadcast!(getindex, y, Ref(x), indices...)
    function _unsafe_tokenselect_pullback(Ybar)
        if isidarray(x)
            dx = NoTangent()
        else
            Ȳ = unthunk(Ybar)
            dx = @thunk begin
                dx = similar(x)
                fill!(dx, zero(eltype(dx)))
                mapreduce(identity, _z,
                          instantiate(broadcasted(∇_getindex!, Ref(dx), Ȳ, Base.OneTo{Int32}(fdim), i, bs...)); init = false)
                return dx
            end
        end
        return (NoTangent(), dx, NoTangent())
    end
    return y, _unsafe_tokenselect_pullback
end
ChainRulesCore.@non_differentiable _unsafe_singletokenselect(x::OneHotArray, i)
function ChainRulesCore.rrule(::typeof(_unsafe_singletokenselect), x::AbstractArray, i)
    @assert isone(size(i, lengthdim(x)))
    y, _unsafe_tokenselect_pullback = rrule(_unsafe_tokenselect, x, i)
    if isidarray(x)
        shape = Base.tail(size(x))
        return reshape(y, shape), _unsafe_tokenselect_pullback
    else
        s = size(y)
        _unsafe_singletokenselect_pullback(Ybar) = _unsafe_tokenselect_pullback(reshape(unthunk(Ybar), s))
        shape = (size(y, 1), Base.tail(Base.tail(size(i)))...)
        return reshape(y, shape), _unsafe_singletokenselect_pullback
    end
end

"""
    lengthselect(x, i)

`selectdim` on the "length" dimension (2 for most array and 1 for integer array).
"""
lengthselect(x::AbstractArray, i) = selectdim(x, lengthdim(x), i)
lengthselect(x::AbstractArray{<:Union{Integer, AbstractOneHotArray}}, i) = ignore_derivatives(()->selectdim(x, lengthdim(x), i))
lengthselect(x::OneHotArray, i) = ignore_derivatives(()->OneHotArray(lengthselect(parent(x), i)))

"""
    skipboundarytoken(x; first=1, last=1)

Select ([`lengthselect`](@ref)) the non-boundary tokens from the hidden states, normally equivalent
 to `x[:, begin+first:end-last, :]`.

See also: [`lengthselect`](@ref)
"""
function skipboundarytoken(x; first = 1, last = 1)
    range = ChainRulesCore.ignore_derivatives() do
        d = lengthdim(x)
        firstidx = firstindex(x, d) + first
        lastidx = lastindex(x, d) - last
        r = firstidx:lastidx
        if isempty(r)
            throw(ArgumentError("Select range is empty: $r"))
        end
        return r
    end
    return lengthselect(x, range)
end

"""
    firsttoken(x)

Slice the first tokens from the hidden states, normally equivalent to `x[:, begin, :]`.

See also: [`lengthselect`](@ref), [`skipboundarytoken`](@ref)
"""
firsttoken(x) = lengthselect(x, lengthdimfirstindex(x))

"""
    firsttoken(x, m::AbstractSeqMask)

Slice the first token from the hidden states. The "first" token is defined by the sequence mask.
"""
firsttoken(x, m::Union{LengthMask, NoMask}) = firsttoken(x)
firsttoken(x, m::AbstractSeqMask) = _unsafe_singletokenselect(x, _findbound(min, x, m))

"""
    lasttoken(x)

Slice the first tokens from the hidden states, normally equivalent to `x[:, end, :]`.

See also: [`lengthselect`](@ref), [`skipboundarytoken`](@ref)
"""
lasttoken(x) = lengthselect(x, lengthdimlastindex(x))

"""
    lasttoken(x, m::AbstractSeqMask)

Slice the last token from the hidden states. The "last" token is defined by the sequence mask.
"""
lasttoken(x, m::Union{RevLengthMask, NoMask}) = lasttoken(x)
lasttoken(x, m::AbstractSeqMask) = _unsafe_singletokenselect(x, _findbound(max, x, m))

"""
    skipfirsttoken(x)

Slice the non-first tokens from the hidden states, normally equivalent to `x[:, 2:end, :]`.

See also: [`lengthselect`](@ref), [`skipboundarytoken`](@ref), [`skiplasttoken`](@ref)
"""
skipfirsttoken(x) = skipboundarytoken(x; first = 1, last = 0)

"""
    skiplasttoken(x)

Slice the non-last tokens from the hidden states, normally equivalent to `x[:, 1:end-1, :]`.

See also: [`lengthselect`](@ref), [`skipboundarytoken`](@ref), [`skipfirsttoken`](@ref)
"""
skiplasttoken(x) = skipboundarytoken(x; first = 0, last = 1)
