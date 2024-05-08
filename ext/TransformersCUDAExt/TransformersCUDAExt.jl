module TransformersCUDAExt

using Transformers
using Transformers.Flux
using CUDA

const FluxCUDAExt = Base.get_extension(Flux, :FluxCUDAExt)

Transformers._toxdevice(adaptor::Flux.FluxCUDAAdaptor, x, cache) =
    Transformers.__toxdevice(adaptor, cache, x, FluxCUDAExt.check_use_cuda)

end
