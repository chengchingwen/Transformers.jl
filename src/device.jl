using Flux
using Flux: GPU_BACKEND, gpu_backend!

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
        return @eval todevice(args...; kws...) = tocpudevice(args...; kws...)
    end
    @static if GPU_BACKEND == "CUDA"
        @eval Main begin
            using CUDA
            CUDA.functional() || error("CUDA not functional")
        end
        @eval todevice(args...; kws...) = tocudadevice(args...; kws...)
    elseif GPU_BACKEND == "AMDGPU"
        @eval Main begin
            using AMDGPU
            AMDGPU.functional() || error("AMDGPU not functional")
        end
        @eval todevice(args...; kws...) = toamdgpudevice(args...; kws...)
    elseif GPU_BACKEND == "Metal"
        @eval Main begin
            using Metal
            Metal.functional() || error("Metal not functional")
        end
        @eval todevice(args...; kws...) = tometaldevice(args...; kws...)
    elseif GPU_BACKEND == "CPU"
        @eval todevice(args...; kws...) = tocpudevice(args...; kws...)
    else
        error("Unsupported GPU backend: $GPU_BACKEND")
    end
end

"""
    todevice(x)

Move data to device, only when gpu is enable with `enable_gpu`, basically equal `Flux.gpu`. Otherwise just `Flux.cpu`.
"""
todevice(args...; kws...) = tocpudevice(args...; kws...)

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

# https://github.com/FluxML/Flux.jl/blob/c442f0ca9ef716dfbc215f2b4422b6c34099f649/src/functor.jl#L182
# https://github.com/FluxML/Flux.jl/blob/c442f0ca9ef716dfbc215f2b4422b6c34099f649/ext/FluxCUDAExt/functor.jl#L56
# https://github.com/FluxML/Flux.jl/blob/c442f0ca9ef716dfbc215f2b4422b6c34099f649/ext/FluxAMDGPUExt/functor.jl#L81
# https://github.com/FluxML/Flux.jl/blob/c442f0ca9ef716dfbc215f2b4422b6c34099f649/ext/FluxMetalExt/functor.jl#L33
@inline __toxdevice__(adaptor, cache, x) = Flux.fmap(Base.Fix1(Flux.adapt, adaptor), x; exclude = Flux._isleaf, cache)
@inline __toxdevice__(adaptor, cache, x, warnf) = (warnf(); __toxdevice__(adaptor, cache, x))
function __toxdevice_generator__(world, source, self, adaptor, cache, x, warnf)
    RT = Core.Compiler.return_type(Flux.adapt, Tuple{adaptor, x}, world)
    body = warnf <: Nothing ?
        Expr(:call, :__toxdevice__, :adaptor, :cache, :x) :
        Expr(:call, :__toxdevice__, :adaptor, :cache, :x, :warnf)
    if isconcretetype(RT)
        body = Expr(:(::), body, RT)
    end
    expr = Expr(:lambda, [Symbol("#self#"), :adaptor, :cache, :x, :warnf],
                Expr(Symbol("scope-block"), Expr(:block, Expr(:return, body))))
    ci = ccall(:jl_expand, Any, (Any, Any), expr, @__MODULE__)
    ci.inlineable = true
    return ci
end
@eval function __toxdevice(adaptor, cache, x, warnf)
    $(Expr(:meta, :generated, __toxdevice_generator__))
    $(Expr(:meta, :generated_only))
end

# overload in extensions
_toxdevice(adaptor::Flux.FluxCPUAdaptor, x, cache) = __toxdevice(adaptor, cache, x, nothing)
