module TransformersAMDGPUExt

using Transformers
using Transformers.Flux
using AMDGPU

const FluxAMDGPUExt = Base.get_extension(Flux, :FluxAMDGPUExt)

# https://github.com/FluxML/Flux.jl/blob/c442f0ca9ef716dfbc215f2b4422b6c34099f649/ext/FluxAMDGPUExt/functor.jl#L81
Transformers._toxdevice(adaptor::Flux.FluxAMDGPUAdaptor, x, cache) =
    Transformers.__toxdevice(adaptor, cache, x, FluxAMDGPUExt._exclude, FluxAMDGPUExt.check_use_amdgpu)

end
