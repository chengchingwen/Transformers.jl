macro isinferred(ex)
    esc(quote
        try
            @inferred $ex
            true
        catch err
            @error err
            isa(err, ErrorException) ? false : rethrow(err)
        end
    end)
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
                    @test @isinferred d(x)
                    @test d(x) ≈ d2(x)
                    @test @isinferred gradient((m, x)->sum(sin.(m(x))), d, x)
                    @test grad[1].W ≈ grad2[1].weight
                    @test isnothing(grad[1].b) || isnothing(grad[1].b) ?
                        grad[1].b == grad2[1].bias : grad[1].b ≈ grad2[1].bias
                    @test grad[2] ≈ grad2[2]
                end
            end
        end
    end

    @testset "fork/nsplit" begin
        for act in (gelu, relu, elu, NNlib.sigmoid_fast, NNlib.tanh_fast, swish, nothing, identity)
            for b in (drandn(Float32, 6), nothing)
                W = drandn(Float32, 6, 4)
                W1 = W[1:3, :]
                W2 = W[4:6, :]
                if isnothing(b)
                    b1 = b2 = b
                else
                    b1 = b[1:3]
                    b2 = b[4:6]
                end
                for x in (drandn(Float32, 4, 3, 2), drandn(Float32, 4, 2), drandn(Float32, 4))
                    ns = Layers.NSplit(2, Layers.Dense(act, W, b))
                    fk = Layers.Fork(Layers.Dense(act, W1, b1), Layers.Dense(act, W2, b2))
                    ns_grad0 = gradient((m, x)->sum(sin.(sum(m(x)))), ns, x)
                    fk_grad0 = gradient((m, x)->sum(sin.(sum(m(x)))), fk, x)
                    ns_grad1 = gradient((m, x)->sum(sin.(m(x)[1])), ns, x)
                    fk_grad1 = gradient((m, x)->sum(sin.(m(x)[1])), fk, x)
                    nsf = ns(x)
                    fkf = fk(x)
                    @test @isinferred ns(x)
                    @test @isinferred fk(x)
                    @test collect(nsf[1]) ≈ collect(fkf[1])
                    @test collect(nsf[2]) ≈ collect(fkf[2])
                    @test @isinferred gradient((m, x)->sum(sin.(sum(m(x)))), ns, x)
                    @test @isinferred gradient((m, x)->sum(sin.(sum(m(x)))), fk, x)
                    @test @isinferred gradient((m, x)->sum(sin.(m(x)[1])), ns, x)
                    @test @isinferred gradient((m, x)->sum(sin.(m(x)[1])), fk, x)
                    @test ns_grad0[1].layer.W[1:3, :] ≈ fk_grad0[1].layers[1].W
                    @test ns_grad0[1].layer.W[4:6, :] ≈ fk_grad0[1].layers[2].W
                    @test ns_grad1[1].layer.W[1:3, :] ≈ fk_grad1[1].layers[1].W
                    @test iszero(@view(ns_grad1[1].layer.W[4:6, :])) && isnothing(fk_grad1[1].layers[2])
                    if !isnothing(b)
                        @test ns_grad0[1].layer.b[1:3] ≈ fk_grad0[1].layers[1].b
                        @test ns_grad0[1].layer.b[4:6] ≈ fk_grad0[1].layers[2].b
                        @test ns_grad1[1].layer.b[1:3] ≈ fk_grad1[1].layers[1].b
                        @test iszero(@view(ns_grad1[1].layer.b[4:6])) && isnothing(fk_grad1[1].layers[2])
                    else
                        @test ns_grad0[1].layer.b == fk_grad0[1].layers[1].b
                        @test ns_grad1[1].layer.b == fk_grad1[1].layers[1].b
                    end
                    @test ns_grad0[2] ≈ fk_grad0[2]
                    @test ns_grad1[2] ≈ fk_grad1[2]
                end
            end
        end
    end

    @testset "transformer" begin
        x = drandn(Float32, 10, 5)
        trf = device(Transformer(Layers.TransformerBlock, 3, 2, 10, 5, 5; collect_outputs = false, return_score = false))
        trf_ws = Layers.WithScore(trf)
        @test @isinferred trf((; hidden_state = x))
        @test @isinferred trf_ws((; hidden_state = x))
        @test @isinferred gradient((m, x)->sum(sin.(m((hidden_state = x,)).hidden_state)), trf, x)
        if VERSION < v"1.9"
            @test_broken @isinferred gradient((m, x)->sum(sin.(m((hidden_state = x,)).hidden_state)), trf_ws, x)
        else
            @test @isinferred gradient((m, x)->sum(sin.(m((hidden_state = x,)).hidden_state)), trf_ws, x)
        end
        trf = device(Transformer(Layers.TransformerBlock, 3, 2, 10, 5, 5; collect_outputs = true, return_score = false))
        trf_ws = Layers.WithScore(trf)
        @test @isinferred trf((; hidden_state = x))
        @test @isinferred trf_ws((; hidden_state = x))
        @test @isinferred gradient((m, x)->sum(sin.(m((hidden_state = x,)).hidden_state)), trf, x)
        if VERSION < v"1.9"
            @test_broken @isinferred gradient((m, x)->sum(sin.(m((hidden_state = x,)).hidden_state)), trf_ws, x)
        else
            @test @isinferred gradient((m, x)->sum(sin.(m((hidden_state = x,)).hidden_state)), trf_ws, x)
        end
        trf = device(Transformer(Layers.TransformerDecoderBlock, 3, 2, 10, 5, 5;
                                 collect_outputs = false, return_score = false))
        trf_ws = Layers.WithScore(trf)
        @test @isinferred trf((; hidden_state = x, memory = x))
        @test @isinferred trf_ws((; hidden_state = x, memory = x))
        @test @isinferred gradient((m, x)->sum(sin.(m((hidden_state = x, memory = x)).hidden_state)), trf, x)
        if VERSION < v"1.9"
            @test_broken @isinferred gradient((m, x)->sum(sin.(m((hidden_state = x, memory = x)).hidden_state)), trf_ws, x)
        else
            @test @isinferred gradient((m, x)->sum(sin.(m((hidden_state = x, memory = x)).hidden_state)), trf_ws, x)
        end
        trf = device(Transformer(Layers.TransformerDecoderBlock, 3, 2, 10, 5, 5;
                                 collect_outputs = true, return_score = false))
        trf_ws = Layers.WithScore(trf)
        @test @isinferred trf((; hidden_state = x, memory = x))
        @test @isinferred trf_ws((; hidden_state = x, memory = x))
        @test @isinferred gradient((m, x)->sum(sin.(m((hidden_state = x, memory = x)).hidden_state)), trf, x)
        if VERSION < v"1.9"
            @test_broken @isinferred gradient((m, x)->sum(sin.(m((hidden_state = x, memory = x)).hidden_state)), trf_ws, x)
        else
            @test @isinferred gradient((m, x)->sum(sin.(m((hidden_state = x, memory = x)).hidden_state)), trf_ws, x)
        end
    end
end
