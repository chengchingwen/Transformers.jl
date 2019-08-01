import Pkg
if VERSION >= v"1.1.0"
    function pkg_version(_module::Module)
        Pkg.Types.read_package(joinpath(dirname(dirname(pathof(_module))), "Project.toml")).version
    end
else
    function pkg_version(_module::Module)
        pkg_id = Base.PkgId(_module)
        env = Pkg.Types.Context().env
        pkg_info = Pkg.Types.manifest_info(env, pkg_id.uuid)
        return VersionNumber(pkg_info["version"])
    end
end

if pkg_version(CUDAnative) >= v"2.1.0"
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
else
    function scatter_add!(ys::CuMatrix{T}, us::CuArray{T}, xs::CuArray{Int}) where T
        function kernel!(ys::CuDeviceArray{T}, us::CuDeviceArray{T}, xs)
            xi = threadIdx().x + (blockIdx().x - 1) * blockDim().x

            if xi <= size(ys, 1)
                @inbounds for i = 1:length(xs)
                    ind = Tuple(CartesianIndices(xs)[i])
                    ys[xi, xs[i]] += us[xi, ind...]
                end
            end

            return
        end

        max_threads = 256
        threads = max_threads
        blocks = ceil(Int, size(ys, 1) / threads)
        CuArrays.@cuda blocks=blocks threads=threads kernel!(ys, us, xs)
        return ys
    end

    function scatter_add!(ys::CuArray{T}, us::CuArray{T}, xs::CuArray{<:Tuple}) where T
        function kernel!(ys::CuDeviceArray{T}, us::CuDeviceArray{T}, xs)
            xi = threadIdx().x + (blockIdx().x - 1) * blockDim().x

            if xi <= size(ys, 1)
                @inbounds for i = 1:length(xs)
                    ind = Tuple(CartesianIndices(xs)[i])
                    ys[xi, xs[i]...] += us[xi, ind...]
                end
            end

            return
        end

        max_threads = 256
        threads = max_threads
        blocks = ceil(Int, size(ys, 1) / threads)
        CuArrays.@cuda blocks=blocks threads=threads kernel!(ys, us, xs)
        return ys
    end
end
