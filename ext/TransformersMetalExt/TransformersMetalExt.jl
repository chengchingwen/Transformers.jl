module TransformersMetalExt

using Transformers
using Transformers.Flux
using Metal

const FluxMetalExt = Base.get_extension(Flux, :FluxMetalExt)

# https://github.com/FluxML/Flux.jl/blob/c442f0ca9ef716dfbc215f2b4422b6c34099f649/ext/FluxMetalExt/functor.jl#L33
Transformers._toxdevice(adaptor::Flux.FluxMetalAdaptor, x, cache) =
    Transformers.__toxdevice(adaptor, cache, x, Flux._isleaf, FluxMetalExt.check_use_metal)

end
