using Statistics
import Statistics: centralize_sumabs2!

Statistics.var(xs::CuArray; corrected::Bool=true, dims=:, mean=mean(xs, dims=dims)) = Statistics.varm(xs, mean; corrected=corrected, dims=dims)
Statistics.std(xs::CuArray; corrected::Bool=true, dims=:, mean=mean(xs, dims=dims)) = sqrt.(var(xs, corrected=corrected, dims=dims, mean=mean))

function Statistics.centralize_sumabs2!(R::CuArray{S}, A::CuArray, means::CuArray) where S
    indsAt, indsRt = Base.safe_tail(axes(A)), Base.safe_tail(axes(R))
    keep, Idefault = Broadcast.shapeindexer(indsRt)
    len = length(CartesianIndices(indsAt))

    function kernel1!(r::CuDeviceArray{S}, a::CuDeviceArray, m::CuDeviceArray, i1)
        li = threadIdx().y + (blockIdx().y - 1) * blockDim().y
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

        @inbounds if li <= len && i <= size(a, 1)
            ia = CartesianIndices(indsAt)[li]
            ir = Tuple(Broadcast.newindex(ia, keep, Idefault))
            x = abs2(a[i, Tuple(ia)...] - m[i1, ir...])
            CUDAnative.atomic_add!(
                pointer(r,
                        Base._to_linear_index(r,
                                              i1,
                                              ir...
                                              )
                        ),
                x
            )
        end
        return
    end

    function kernel!(r::CuDeviceArray{S}, a::CuDeviceArray, m::CuDeviceArray)
        li = threadIdx().y + (blockIdx().y - 1) * blockDim().y
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

        @inbounds if li <= len && i <= size(a, 1)
            ia = CartesianIndices(indsAt)[li]
            ir = Tuple(Broadcast.newindex(ia, keep, Idefault))
            x = abs2(a[i, Tuple(ia)...] - m[i, ir...])
            CUDAnative.atomic_add!(
                pointer(r,
                        Base._to_linear_index(r,
                                              i,
                                              ir...
                                              )
                        ),
                x
            )
        end
        return
    end

    max_threads = 256
    thread_x = min(max_threads, size(A, 1))
    thread_y = min(max_threads ÷ thread_x, len)
    threads = (thread_x, thread_y)
    blocks = ceil.(Int, (size(A, 1), len) ./ threads)

    if Base.reducedim1(R, A)
        i1 = first(Base.axes1(R))
        CuArrays.@cuda blocks=blocks threads=threads kernel1!(R, A, means, i1)
    else
        CuArrays.@cuda blocks=blocks threads=threads kernel!(R, A, means)
    end

    return R
end
