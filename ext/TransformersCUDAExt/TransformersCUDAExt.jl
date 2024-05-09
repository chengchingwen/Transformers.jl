module TransformersCUDAExt

using Transformers
using Transformers.Flux
using CUDA

const FluxCUDAExt = Base.get_extension(Flux, :FluxCUDAExt)

# https://github.com/FluxML/Flux.jl/blob/c442f0ca9ef716dfbc215f2b4422b6c34099f649/ext/FluxCUDAExt/functor.jl#L56
Transformers._toxdevice(adaptor::Flux.FluxCUDAAdaptor, x, cache) =
    Transformers.__toxdevice(adaptor, cache, x, Flux._isleaf, FluxCUDAExt.check_use_cuda)

end
