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

@init @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
    using .CuArrays
    using .CuArrays.CUDAnative

    # gpu gather
    gather(w::CuMatrix{T}, xs::OneHotArray) where T = gather(w, onehot2indices(xs))
    function gather(w::CuMatrix{T}, xs::CuArray{Int}) where T
        ys = CuArray{T}(undef, size(w, 1), size(xs)...)

        function kernel!(ys::CuDeviceArray{T}, w::CuDeviceArray{T}, xs)
            li = threadIdx().y + (blockIdx().y - 1) * blockDim().y
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

            @inbounds if li <= length(xs)
                ind = Tuple(CartesianIndices(xs)[li])
                ys[i, ind...] = w[i, xs[li]]
            end

            return
        end

        max_threads = 256
        threads_x = min(max_threads, size(ys,1))
        threads_y = min(max_threads ÷ threads_x, length(xs))
        threads = (threads_x, threads_y)
        blocks = ceil.(Int, (size(ys,1), length(xs)) ./ threads)

        CuArrays.@cuda blocks=blocks threads=threads kernel!(ys, w, xs)
        return ys
    end

    function gather(W::CuArray{T}, xs::CuArray{<:Tuple}) where T
        ys = CuArray{T}(undef, size(W, 1), size(xs)...)

        function kernel!(ys::CuDeviceArray{T}, w::CuDeviceArray{T}, xs)
            li = threadIdx().y + (blockIdx().y - 1) * blockDim().y
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

            @inbounds if li <= length(xs)
                ind = Tuple(CartesianIndices(xs)[li])
                ys[i, ind...] = w[i, xs[li]...]
            end

            return
        end

        max_threads = 256
        threads_x = min(max_threads, size(ys,1))
        threads_y = min(max_threads ÷ threads_x, length(xs))
        threads = (threads_x, threads_y)
        blocks = ceil.(Int, (size(ys,1), length(xs)) ./ threads)

        CuArrays.@cuda blocks=blocks threads=threads kernel!(ys, W, xs)
        return ys
    end
end

using Flux: Tracker, TrackedArray, TrackedMatrix, data
using Flux.Tracker: @grad, track

gather(w::TrackedMatrix, xs::OneHotArray) = gather(w, onehot2indices(xs))
gather(w::TrackedMatrix, xs::AbstractArray{Int}) = track(gather, w, xs)
gather(w::TrackedArray, xs::AbstractArray{<:Tuple}) = track(gather, w, xs)

@grad gather(w::TrackedArray, xs) = gather(data(w), xs), Δ->(∇gather(Δ, data(w), xs),nothing)
