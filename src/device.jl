using Flux
using Flux: GPU_BACKEND, gpu_backend!
using Functors

is_precompiling() = ccall(:jl_generating_output, Cint, ()) == 1

"""
    enable_gpu(t=true)

Enable gpu for `todevice`, disable with `enable_gpu(false)`. The backend is selected by `Flux.gpu_backend!`.
 Should only be used in user scripts.
"""
function enable_gpu(t::Bool=true)
    if is_precompiling()
        @error """
        `Transformers.enable_gpu` is called during precompilation, result in a no-op.
        `enable_gpu` should only be used in user scripts.
        """
        return todevice
    end
    if !t
        return @eval @inline todevice(args...; kws...) = tocpudevice(args...; kws...)
    end
    @static if GPU_BACKEND == "CUDA"
        @eval Main begin
            using CUDA
            CUDA.functional() || error("CUDA not functional")
        end
    elseif GPU_BACKEND == "AMDGPU"
        @eval Main begin
            using AMDGPU
            AMDGPU.functional() || error("AMDGPU not functional")
        end
    elseif GPU_BACKEND == "Metal"
        @eval Main begin
            using Metal
            Metal.functional() || error("Metal not functional")
        end
    elseif GPU_BACKEND == "CPU"
    else
        error("Unsupported GPU backend: $GPU_BACKEND")
    end
    @eval @inline todevice(args...; kws...) = togpudevice(args...; kws...)
end

"""
    todevice(x)

Move data to device, only when gpu is enable with `enable_gpu`, basically equal `Flux.gpu`. Otherwise just `Flux.cpu`.
"""
@inline todevice(args...; kws...) = tocpudevice(args...; kws...)

"""
    togpudevice(x)

Move data to gpu device, backend selected by `Flux.gpu_backend!`.
"""
@inline function togpudevice(args...; kws...)
    @static if GPU_BACKEND == "CUDA"
        return tocudadevice(args...; kws...)
    elseif GPU_BACKEND == "AMDGPU"
        return toamdgpudevice(args...; kws...)
    elseif GPU_BACKEND == "Metal"
        return tometaldevice(args...; kws...)
    elseif GPU_BACKEND == "CPU"
        return tocpudevice(args...; kws...)
    else
        error("Unsupported GPU backend: $GPU_BACKEND")
    end
end

const FluxAdaptor = Union{Flux.FluxCPUAdaptor, Flux.FluxCUDAAdaptor, Flux.FluxAMDGPUAdaptor, Flux.FluxMetalAdaptor}
tocpudevice(args...; cache = IdDict()) = toxdevice(Flux.FluxCPUAdaptor(), args...; cache)
tocudadevice(args...; cache = IdDict()) = toxdevice(Flux.FluxCUDAAdaptor(), args...; cache)
toamdgpudevice(args...; cache = IdDict()) = toxdevice(Flux.FluxAMDGPUAdaptor(), args...; cache)
tometaldevice(args...; cache = IdDict()) = toxdevice(Flux.FluxMetalAdaptor(), args...; cache)

toxdevice(adaptor::FluxAdaptor, x; cache = IdDict()) = _toxdevice(adaptor, x, cache)
function toxdevice(adaptor::FluxAdaptor, x, xs...; cache = IdDict())
    return (toxdevice(adaptor, x; cache), map(xi->toxdevice(adaptor, xi; cache), xs)...)
end
toxdevice(adaptor::FluxAdaptor, x::Tuple; cache = IdDict()) = toxdevice(adaptor, x...; cache)
toxdevice(adaptor::FluxAdaptor, x::Tuple{Any}; cache = IdDict()) = (toxdevice(adaptor, x...; cache),)
toxdevice(adaptor::FluxAdaptor, x::NamedTuple{name}; cache = IdDict()) where name =
    NamedTuple{name}(toxdevice(adaptor, values(x); cache))

struct AdaptorCache{A, C} <: AbstractDict{Any, Any}
    adaptor::A
    cache::C
end
Base.haskey(cache::AdaptorCache, x) = haskey(cache.cache, x)
Base.iterate(cache::AdaptorCache, state...) = iterate(cache.cache, state...)
Base.setindex!(cache::AdaptorCache, value, key) = setindex!(cache.cache, value, key)
function __cacheget_generator__(world, source, self, cache, x)
    adaptor = cache.parameters[1]
    RT = Core.Compiler.return_type(Flux.adapt, Tuple{adaptor, x}, world)
    body = Expr(:call, GlobalRef(Base, :getindex), Expr(:., :cache, QuoteNode(:cache)), :x)
    body = Expr(:(::), body, RT)
    expr = Expr(:lambda, [Symbol("#self#"), :cache, :x],
                Expr(Symbol("scope-block"), Expr(:block, Expr(:return, body))))
    ci = ccall(:jl_expand, Any, (Any, Any), expr, @__MODULE__)
    ci.inlineable = true
    return ci
end
@eval function Base.getindex(cache::AdaptorCache, x)
    $(Expr(:meta, :generated, __cacheget_generator__))
    $(Expr(:meta, :generated_only))
end
# https://github.com/FluxML/Functors.jl/blob/cfc6a608e309c64e4da0f44cd937cb9efa4fd6c7/src/walks.jl#L190
# CachedWalk + AdaptorCache: CachedWalk only take cache::IdDict, so we made our own
struct AdaptorWalk{W<:Functors.AbstractWalk, C<:AdaptorCache} <: Functors.AbstractWalk
    walk::W
    cache::C
end
function (walk::AdaptorWalk)(recurse, x, ys...)
    should_cache = Functors.usecache(walk.cache, x)
    if should_cache && haskey(walk.cache, x)
        return walk.cache[x]
    else
        ret = walk.walk(recurse, x, ys...)
        if should_cache
            walk.cache[x] = ret
        end
        return ret
    end
end

# https://github.com/FluxML/Flux.jl/blob/c442f0ca9ef716dfbc215f2b4422b6c34099f649/src/functor.jl#L182
# https://github.com/FluxML/Flux.jl/blob/c442f0ca9ef716dfbc215f2b4422b6c34099f649/ext/FluxCUDAExt/functor.jl#L56
# https://github.com/FluxML/Flux.jl/blob/c442f0ca9ef716dfbc215f2b4422b6c34099f649/ext/FluxAMDGPUExt/functor.jl#L81
# https://github.com/FluxML/Flux.jl/blob/c442f0ca9ef716dfbc215f2b4422b6c34099f649/ext/FluxMetalExt/functor.jl#L33
# https://github.com/FluxML/Functors.jl/blob/cfc6a608e309c64e4da0f44cd937cb9efa4fd6c7/src/maps.jl#L11
@inline function __toxdevice(adaptor, cache, x, exclude, warnf)
    !isnothing(warnf) && warnf()
    walk = Functors.ExcludeWalk(Functors.DefaultWalk(), Base.Fix1(Flux.adapt, adaptor), exclude)
    walk = AdaptorWalk(walk, AdaptorCache(adaptor, cache))
    return Functors.execute(walk, x)
end

# overload in extensions
_toxdevice(adaptor::Flux.FluxCPUAdaptor, x, cache) = __toxdevice(adaptor, cache, x, Flux._isleaf, nothing)
