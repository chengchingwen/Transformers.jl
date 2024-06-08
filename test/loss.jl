@testset "Loss" begin
    using Statistics
    using PrimitiveOneHot
    using Flux
    using Flux.Losses
    using ChainRulesCore
    using NeuralAttentionlib: LengthMask, RevLengthMask, GenericSeqMask, getmask, lengths
    using Transformers: safe_crossentropy, safe_logitcrossentropy

    noagg(x; dims = :) = x
    _crossentropy(args...; kwargs...) = crossentropy(args...; kwargs..., eps = 0f0)
    function naive_loss_w_mask(ŷ, y, m; lossf = _crossentropy, agg = mean)
        batch = size(ŷ, ndims(ŷ))
        loss = lossf(reshape(ŷ, size(ŷ,1), :), reshape(y, size(y,1), :); dims = 1, agg = noagg)
        loss = reshape(loss, 1, Base.tail(size(ŷ))...)
        dims = ntuple(i->i+1, ndims(loss) - 2)
        loss = reshape(sum(loss .* m; dims), batch)
        loss = sum(loss ./ lengths(m))
        if agg isa typeof(mean)
            return loss / batch
        end
        return loss
    end
    max_length = 50
    nclass = 500

    for i = 1:3
        if i == 1
            x = drandn(Float32, nclass, max_length, 2)
            l = OneHotArray(nclass, drand(1:nclass, max_length, 2))
            m0 = drand(Bool, max_length, 2)
            m1 = drand(1:max_length, 2)
            m2 = drand(1:max_length, 2)
        elseif i == 2
            x = drandn(Float32, nclass, max_length, 2, 2)
            l = OneHotArray(nclass, drand(1:nclass, max_length, 2, 2))
            m0 = drand(Bool, max_length, 2, 2)
            m1 = drand(1:max_length, 2, 2)
            m2 = drand(1:max_length, 2, 2)
        else
            x = drandn(Float32, nclass, max_length, 1)
            l = OneHotArray(nclass, drand(1:nclass, max_length, 1))
            m0 = drand(Bool, max_length, 1)
            m1 = drand(1:max_length, 1)
            m2 = drand(1:max_length, 1)
        end
        z = softmax(x; dims=1)
        l0 = device(collect(l))
        li = reinterpret(Int32, l)
        M0 = GenericSeqMask(m0)
        M1 = LengthMask(m1)
        M2 = RevLengthMask(m1)

        for M in (M0, M1, M2)
            @test crossentropy(sum, z, l, M; ϵ = 1f-10) ≈ naive_loss_w_mask(z, l, M; agg = sum)
            @test crossentropy(mean, z, l, M; ϵ = 1f-10) ≈ naive_loss_w_mask(z, l, M; agg = mean)
            @test crossentropy(sum, z, l0, M; ϵ = 1f-10) ≈ naive_loss_w_mask(z, l0, M; agg = sum)
            @test crossentropy(mean, z, l0, M; ϵ = 1f-10) ≈ naive_loss_w_mask(z, l0, M; agg = mean)
            @test safe_crossentropy(sum, z, li, M; ϵ = 1f-10) ≈ naive_loss_w_mask(z, l, M; agg = sum)
            @test safe_crossentropy(mean, z, li, M; ϵ = 1f-10) ≈ naive_loss_w_mask(z, l, M; agg = mean)

            @test Flux.gradient(z->crossentropy(sum, z, l, M; ϵ = 1f-10), z)[1] ≈
                Flux.gradient(z->naive_loss_w_mask(z, l, M; agg = sum), z)[1]
            @test Flux.gradient(z->crossentropy(mean, z, l, M; ϵ = 1f-10), z)[1] ≈
                Flux.gradient(z->naive_loss_w_mask(z, l, M; agg = mean), z)[1]
            @test Flux.gradient(z->crossentropy(sum, z, l0, M; ϵ = 1f-10), z)[1] ≈
                Flux.gradient(z->naive_loss_w_mask(z, l0, M; agg = sum), z)[1]
            @test Flux.gradient(z->crossentropy(mean, z, l0, M; ϵ = 1f-10), z)[1] ≈
                Flux.gradient(z->naive_loss_w_mask(z, l0, M; agg = mean), z)[1]
            @test Flux.gradient(z->safe_crossentropy(sum, z, li, M; ϵ = 1f-10), z)[1] ≈
                Flux.gradient(z->naive_loss_w_mask(z, l, M; agg = sum), z)[1]
            @test Flux.gradient(z->safe_crossentropy(mean, z, li, M; ϵ = 1f-10), z)[1] ≈
                Flux.gradient(z->naive_loss_w_mask(z, l, M; agg = mean), z)[1]

            @test logitcrossentropy(sum, x, l, M) ≈
                naive_loss_w_mask(x, l, M; lossf = logitcrossentropy, agg = sum)
            @test logitcrossentropy(mean, x, l, M) ≈
                naive_loss_w_mask(x, l, M; lossf = logitcrossentropy, agg = mean)
            @test logitcrossentropy(sum, x, l0, M) ≈
                naive_loss_w_mask(x, l0, M; lossf = logitcrossentropy, agg = sum)
            @test logitcrossentropy(mean, x, l0, M) ≈
                naive_loss_w_mask(x, l0, M; lossf = logitcrossentropy, agg = mean)
            @test safe_logitcrossentropy(sum, x, li, M) ≈
                naive_loss_w_mask(x, l, M; lossf = logitcrossentropy, agg = sum)
            @test safe_logitcrossentropy(mean, x, li, M) ≈
                naive_loss_w_mask(x, l, M; lossf = logitcrossentropy, agg = mean)

            @test Flux.gradient(x->logitcrossentropy(sum, x, l, M), x)[1] ≈
                Flux.gradient(x->naive_loss_w_mask(x, l, M; lossf = logitcrossentropy, agg = sum), x)[1]
            @test Flux.gradient(x->logitcrossentropy(mean, x, l, M), x)[1] ≈
                Flux.gradient(x->naive_loss_w_mask(x, l, M; lossf = logitcrossentropy, agg = mean), x)[1]
            @test Flux.gradient(x->logitcrossentropy(sum, x, l0, M), x)[1] ≈
                Flux.gradient(x->naive_loss_w_mask(x, l0, M; lossf = logitcrossentropy, agg = sum), x)[1]
            @test Flux.gradient(x->logitcrossentropy(mean, x, l0, M), x)[1] ≈
                Flux.gradient(x->naive_loss_w_mask(x, l0, M; lossf = logitcrossentropy, agg = mean), x)[1]
            @test Flux.gradient(x->safe_logitcrossentropy(sum, x, li, M), x)[1] ≈
                Flux.gradient(x->naive_loss_w_mask(x, l, M; lossf = logitcrossentropy, agg = sum), x)[1]
            @test Flux.gradient(x->safe_logitcrossentropy(mean, x, li, M), x)[1] ≈
                Flux.gradient(x->naive_loss_w_mask(x, l, M; lossf = logitcrossentropy, agg = mean), x)[1]
        end
    end
end
