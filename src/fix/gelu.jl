import Flux: gelu

@init @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
    import .CuArrays
    using .CuArrays: CuArray
    using .CuArrays.CUDAnative

    CuArrays.@cufunc function gelu(x)
        λ = oftype(x/1, √(2/π))
        α = oftype(x/1, 0.044715)
        h = oftype(x/1, 0.5)
        h * x * (one(x) + tanh(λ * (x + α * x^3)))
    end
end
