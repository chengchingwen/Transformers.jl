using CuArrays
using CUDAnative

function Base.one(xs::CuMatrix{T}) where T
    n, n2 = size(xs)
    if n != n2
        throw(DimensionMismatch("multiplicative identity defined only for square matrices"))
    end

    ys = CuArray{T}(undef, n, n)
    fill!(ys, zero(T))

    num_threads = min(n, 256)
    num_blocks = ceil(Int, n / num_threads)

    function kernel(ys::CuArrays.CuDeviceArray{T})
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        if i <= size(ys, 1)
            ys[i,i] = one(T)
        end

        return
    end

    @cuda blocks=num_blocks threads=num_threads kernel(ys)

    return ys
end


