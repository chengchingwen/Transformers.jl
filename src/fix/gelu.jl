import Flux: gelu
using CuArrays
using CUDAnative

CuArrays.@cufunc function gelu(x)
    λ = oftype(x/1, √(2/π))
    α = oftype(x/1, 0.044715)
    h = oftype(x/1, 0.5)
    h * x * (one(x) + tanh(λ * (x + α * x^3)))
end

