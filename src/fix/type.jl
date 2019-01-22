using Flux
import Flux: Dense, Diagonal, LayerNorm, glorot_uniform, glorot_normal

glorot_uniform(T::Type, dims...) = (rand(T, dims...) .- T(0.5)) .* sqrt(T(24.0)/sum(dims))
glorot_normal(T::Type, dims...) = randn(T, dims...) .* sqrt(T(2.0)/sum(dims))

function Dense(dtype::Type, in::Integer, out::Integer, σ = identity;
               initW = glorot_uniform, initb = zeros)
  return Dense(param(initW(dtype, out, in)), param(initb(dtype, out)), σ)
end

Diagonal(dtype::Type, in::Integer; initα = ones, initβ = zeros) =
  Diagonal(param(initα(dtype, in)), param(initβ(dtype, in)))

LayerNorm(dtype::Type, h::Integer) =
  LayerNorm(Diagonal(dtype, h))

