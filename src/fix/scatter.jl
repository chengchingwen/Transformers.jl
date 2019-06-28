for op = [:add, :sub, :max, :min]
    @eval function $(Symbol("scatter_", op, "!"))(ys::Matrix{T}, us::Array{T}, xs::Array{Int}) where T
        l = length(xs)
        s = size(ys, 1)

        Threads.@threads for num ∈ 1:(l*s)
            li = (num -1) ÷ s + 1
            i = (num - 1) % s + 1
            @inbounds ind = Tuple(CartesianIndices(xs)[li])
            @inbounds $(Symbol("atomic_", op, "!"))(
                pointer(ys,
                        Base._to_linear_index(ys,
                                              i,
                                              xs[li]
                                              )
                        ),
                us[i, ind...]
            )
        end

        return ys
    end

    @eval function $(Symbol("scatter_", op, "!"))(ys::Array{T}, us::Array{T}, xs::Array{<:Tuple}) where T
        l = length(xs)
        s = size(ys, 1)

        Threads.@threads for num = 1:(l*s)
            li = (num -1) ÷ s + 1
            i = (num - 1) % s + 1
            @inbounds ind = Tuple(CartesianIndices(xs)[li])
            @inbounds $(Symbol("atomic_", op, "!"))(
                pointer(ys,
                        Base._to_linear_index(ys,
                                              i,
                                              xs[li]...
                                              )
                        ),
                us[i, ind...]
            )
        end

        return ys
    end
end

@init @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
    using .CuArrays
    using .CuArrays.CUDAnative
    include("./scatter_gpu.jl")
end
