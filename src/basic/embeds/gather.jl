"
    gather(w::AbstractMatrix{T}, xs::OneHotArray) where

getting vector at the given onehot encoding.
"
gather(w::AbstractMatrix{T}, xs::OneHotArray) where T = gather(w, onehot2indices(xs))

"
    gather(w::AbstractMatrix{T}, xs) where

getting vector at the given indices, `xs` is a array of indices(`Int` type).
"
function gather(w::AbstractMatrix{T}, xs::AbstractArray{Int}) where T
    ys = similar(w, size(w, 1), size(xs)...)

    Threads.@threads for i = 1:length(xs)
        @inbounds ind = Tuple(CartesianIndices(xs)[i])
        @inbounds ys[:, ind...] .= w[:, xs[i]]
    end
    return ys
end

"
    gather(w::AbstractArray{T}, xs) where

getting vector at the given indices, `xs` is a array of cartesian indices(`Tuple{Int}` type).
"
function gather(w::AbstractArray{T}, xs::AbstractArray{<:Tuple}) where T
    ys = similar(w, size(w, 1), size(xs)...)

    Threads.@threads for i = 1:length(xs)
        @inbounds ind = Tuple(CartesianIndices(xs)[i])
        @inbounds ys[:, ind...] .= w[:, xs[i]...]
    end
    return ys
end

# cpu gather back
function ∇gather(Δ::AbstractArray{T}, w::AbstractMatrix{T}, xs::AbstractArray{Int}) where T
    ys = fill!(similar(w), zero(T))
    scatter_add!(ys, Δ, xs)
    return ys
end

function ∇gather(Δ::AbstractArray{T}, w::AbstractArray{T}, xs::AbstractArray{<:Tuple}) where T
    ys = fill!(similar(w), zero(T))
    scatter_add!(ys, Δ, xs)
    return ys
end

using ZygoteRules: @adjoint, AContext, pullback
import ZygoteRules: _pullback

_pullback(cx::AContext, ::typeof(gather), w, xs::OneHotArray) = _pullback(cx, gather, w, onehot2indices(xs))
@adjoint gather(w, xs::AbstractArray{<: Union{Int, <:Tuple}}) = gather(w, xs), Δ->(∇gather((similar(w, size(Δ)) .= Δ), w, xs),nothing)
