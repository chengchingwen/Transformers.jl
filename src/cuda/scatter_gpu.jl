for op = [:add, # :sub, :max, :min,
          ]
    @eval function $(Symbol("scatter_", op, "!"))(ys::CuMatrix{T}, us::CuArray{T}, xs::CuArray{Int}) where T
        function kernel!(ys, us, xs)
            li = threadIdx().y + (blockIdx().y - 1) * blockDim().y
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

            @inbounds if li <= length(xs) && i <= size(ys, 1)
                ind = Tuple(CartesianIndices(xs)[li])
                CUDAnative.$(Symbol("atomic_", op, "!"))(
                    pointer(ys,
                            Base._to_linear_index(ys,
                                                  i, xs[li]
                                                  )
                            ),
                    us[i, ind...]
                )
            end

            return
        end

        max_threads = 256
        thread_x = min(max_threads, size(ys, 1))
        thread_y = min(max_threads ÷ thread_x, length(xs))
        threads = (thread_x, thread_y)
        blocks = ceil.(Int, (size(ys, 1), length(xs)) ./ threads)
        @cuda blocks=blocks threads=threads kernel!(ys, us, xs)
        return ys
    end

    @eval function $(Symbol("scatter_", op, "!"))(ys::CuArray{T}, us::CuArray{T}, xs::CuArray{<:Tuple}) where T
        function kernel!(ys, us, xs)
            li = threadIdx().y + (blockIdx().y - 1) * blockDim().y
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

            @inbounds if li <= length(xs) && i <= size(ys, 1)
                ind = Tuple(CartesianIndices(xs)[li])
                CUDAnative.$(Symbol("atomic_", op, "!"))(
                    pointer(ys,
                            Base._to_linear_index(ys,
                                                  i, xs[li]...
                                                  )
                            ),
                    us[i, ind...]
                )
            end

            return
        end

        max_threads = 256
        thread_x = min(max_threads, size(ys, 1))
        thread_y = min(max_threads ÷ thread_x, length(xs))
        threads = (thread_x, thread_y)
        blocks = ceil.(Int, (size(ys, 1), length(xs)) ./ threads)
        @cuda blocks=blocks threads=threads kernel!(ys, us, xs)
        return ys
    end
end
