module TransformersAMDGPUExt.jl

using Transformers
using Transformers.Flux
using AMDGPU

const FluxAMDGPUExt = Base.get_extension(Flux, :FluxAMDGPUExt)

Transformers._toxdevice(adaptor::Flux.FluxAMDGPUAdaptor, x, cache) =
    Transformers.__toxdevice(adaptor, cache, x, FluxAMDGPUExt.check_use_amdgpu)

end
