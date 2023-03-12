macro isinferred(ex)
    quote
        try
            f = () -> $(esc(ex))
            @inferred f()
            true
        catch err
            isa(err, ErrorException) ? false : rethrow(err)
        end
    end
end

@testset "Grad" begin
    using NNlib
    using Flux
    using Flux.Zygote
    @testset "dense" begin
        for act in (gelu, relu, elu, NNlib.sigmoid_fast, NNlib.tanh_fast, swish, nothing, identity)
            for b in (drandn(Float32, 5), nothing)
                W = drandn(Float32, 5, 4)
                for x in (drandn(Float32, 4, 3, 2), drandn(Float32, 4, 2), drandn(Float32, 4))
                    d = Layers.Dense(act, W, b)
                    d2 = Dense(W, isnothing(b) ? false : b, isnothing(act) ? identity : act)
                    grad = gradient((m, x)->sum(sin.(m(x))), d, x)
                    grad2 = gradient((m, x)->sum(sin.(m(x))), d2, x)
                    @test @isinferred gradient((m, x)->sum(sin.(m(x))), d2, x)
                    @test grad[1].W ≈ grad2[1].weight
                    @test isnothing(grad[1].b) || isnothing(grad[1].b) ?
                        grad[1].b == grad2[1].bias : grad[1].b ≈ grad2[1].bias
                    @test grad[2] ≈ grad2[2]
                end
            end
        end
    end
end
