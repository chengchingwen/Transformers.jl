import .Basic: gather

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

    CUDA.@cuda blocks=blocks threads=threads kernel!(ys, w, xs)
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

    CUDA.@cuda blocks=blocks threads=threads kernel!(ys, W, xs)
    return ys
end
