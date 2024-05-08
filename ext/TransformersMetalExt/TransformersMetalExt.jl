module TransformersMetalExt

using Transformers
using Transformers.Flux
using Metal

const FluxMetalExt = Base.get_extension(Flux, :FluxMetalExt)

Transformers._toxdevice(adaptor::Flux.FluxMetalAdaptor, x, cache) =
    Transformers.__toxdevice(adaptor, cache, x, FluxMetalExt.check_use_metal)

end
