@testset "Loss" begin
    using Statistics
    using PrimitiveOneHot
    using Flux
    using Flux.Losses
    using ChainRulesCore
    using NeuralAttentionlib: LengthMask, RevLengthMask, GenericSequenceMask, getmask, lengths

    noagg(x; dims = :) = x
    _crossentropy(args...; kwargs...) = crossentropy(args...; kwargs..., ϵ = 0f0)
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
            z = softmax(x; dims=1)
            l = OneHotArray(nclass, drand(1:nclass, max_length, 2))
            l0 = device(collect(l))
            m0 = drand(Bool, max_length, 2)
            M0 = GenericSequenceMask(m0)
            m1 = drand(1:max_length, 2)
            M1 = LengthMask(m1)
            m2 = drand(1:max_length, 2)
            M2 = RevLengthMask(m1)
        elseif i == 2
            x = drandn(Float32, nclass, max_length, 2, 2)
            z = softmax(x; dims=1)
            l = OneHotArray(nclass, drand(1:nclass, max_length, 2, 2))
            l0 = device(collect(l))
            m0 = drand(Bool, max_length, 2, 2)
            M0 = GenericSequenceMask(m0)
            m1 = drand(1:max_length, 2, 2)
            M1 = LengthMask(m1)
            m2 = drand(1:max_length, 2, 2)
            M2 = RevLengthMask(m1)
        else
            x = drandn(Float32, nclass, max_length, 1)
            z = softmax(x; dims=1)
            l = OneHotArray(nclass, drand(1:nclass, max_length, 1))
            l0 = device(collect(l))
            m0 = drand(Bool, max_length, 1)
            M0 = GenericSequenceMask(m0)
            m1 = drand(1:max_length, 1)
            M1 = LengthMask(m1)
            m2 = drand(1:max_length, 1)
            M2 = RevLengthMask(m1)
        end

        @test crossentropy(sum, z, l, M0; ϵ = 1f-10) ≈ naive_loss_w_mask(z, l, M0; agg = sum)
        @test crossentropy(mean, z, l, M0; ϵ = 1f-10) ≈ naive_loss_w_mask(z, l, M0; agg = mean)
        @test crossentropy(sum, z, l0, M0; ϵ = 1f-10) ≈ naive_loss_w_mask(z, l0, M0; agg = sum)
        @test crossentropy(mean, z, l0, M0; ϵ = 1f-10) ≈ naive_loss_w_mask(z, l0, M0; agg = mean)
        @test crossentropy(sum, z, l, M1; ϵ = 1f-10) ≈ naive_loss_w_mask(z, l, M1; agg = sum)
        @test crossentropy(mean, z, l, M1; ϵ = 1f-10) ≈ naive_loss_w_mask(z, l, M1; agg = mean)
        @test crossentropy(sum, z, l0, M1; ϵ = 1f-10) ≈ naive_loss_w_mask(z, l0, M1; agg = sum)
        @test crossentropy(mean, z, l0, M1; ϵ = 1f-10) ≈ naive_loss_w_mask(z, l0, M1; agg = mean)
        @test crossentropy(sum, z, l, M2; ϵ = 1f-10) ≈ naive_loss_w_mask(z, l, M2; agg = sum)
        @test crossentropy(mean, z, l, M2; ϵ = 1f-10) ≈ naive_loss_w_mask(z, l, M2; agg = mean)
        @test crossentropy(sum, z, l0, M2; ϵ = 1f-10) ≈ naive_loss_w_mask(z, l0, M2; agg = sum)
        @test crossentropy(mean, z, l0, M2; ϵ = 1f-10) ≈ naive_loss_w_mask(z, l0, M2; agg = mean)

        @test Flux.gradient(z->crossentropy(sum, z, l, M0; ϵ = 1f-10), z)[1] ≈
            Flux.gradient(z->naive_loss_w_mask(z, l, M0; agg = sum), z)[1]
        @test Flux.gradient(z->crossentropy(mean, z, l, M0; ϵ = 1f-10), z)[1] ≈
            Flux.gradient(z->naive_loss_w_mask(z, l, M0; agg = mean), z)[1]
        @test Flux.gradient(z->crossentropy(sum, z, l0, M0; ϵ = 1f-10), z)[1] ≈
            Flux.gradient(z->naive_loss_w_mask(z, l0, M0; agg = sum), z)[1]
        @test Flux.gradient(z->crossentropy(mean, z, l0, M0; ϵ = 1f-10), z)[1] ≈
            Flux.gradient(z->naive_loss_w_mask(z, l0, M0; agg = mean), z)[1]
        @test Flux.gradient(z->crossentropy(sum, z, l, M1; ϵ = 1f-10), z)[1] ≈
            Flux.gradient(z->naive_loss_w_mask(z, l, M1; agg = sum), z)[1]
        @test Flux.gradient(z->crossentropy(mean, z, l, M1; ϵ = 1f-10), z)[1] ≈
            Flux.gradient(z->naive_loss_w_mask(z, l, M1; agg = mean), z)[1]
        @test Flux.gradient(z->crossentropy(sum, z, l0, M1; ϵ = 1f-10), z)[1] ≈
            Flux.gradient(z->naive_loss_w_mask(z, l0, M1; agg = sum), z)[1]
        @test Flux.gradient(z->crossentropy(mean, z, l0, M1; ϵ = 1f-10), z)[1] ≈
            Flux.gradient(z->naive_loss_w_mask(z, l0, M1; agg = mean), z)[1]
        @test Flux.gradient(z->crossentropy(sum, z, l, M2; ϵ = 1f-10), z)[1] ≈
            Flux.gradient(z->naive_loss_w_mask(z, l, M2; agg = sum), z)[1]
        @test Flux.gradient(z->crossentropy(mean, z, l, M2; ϵ = 1f-10), z)[1] ≈
            Flux.gradient(z->naive_loss_w_mask(z, l, M2; agg = mean), z)[1]
        @test Flux.gradient(z->crossentropy(sum, z, l0, M2; ϵ = 1f-10), z)[1] ≈
            Flux.gradient(z->naive_loss_w_mask(z, l0, M2; agg = sum), z)[1]
        @test Flux.gradient(z->crossentropy(mean, z, l0, M2; ϵ = 1f-10), z)[1] ≈
            Flux.gradient(z->naive_loss_w_mask(z, l0, M2; agg = mean), z)[1]

        @test logitcrossentropy(sum, x, l, M0) ≈
            naive_loss_w_mask(x, l, M0; lossf = logitcrossentropy, agg = sum)
        @test logitcrossentropy(mean, x, l, M0) ≈
            naive_loss_w_mask(x, l, M0; lossf = logitcrossentropy, agg = mean)
        @test logitcrossentropy(sum, x, l0, M0) ≈
            naive_loss_w_mask(x, l0, M0; lossf = logitcrossentropy, agg = sum)
        @test logitcrossentropy(mean, x, l0, M0) ≈
            naive_loss_w_mask(x, l0, M0; lossf = logitcrossentropy, agg = mean)
        @test logitcrossentropy(sum, x, l, M1) ≈
            naive_loss_w_mask(x, l, M1; lossf = logitcrossentropy, agg = sum)
        @test logitcrossentropy(mean, x, l, M1) ≈
            naive_loss_w_mask(x, l, M1; lossf = logitcrossentropy, agg = mean)
        @test logitcrossentropy(sum, x, l0, M1) ≈
            naive_loss_w_mask(x, l0, M1; lossf = logitcrossentropy, agg = sum)
        @test logitcrossentropy(mean, x, l0, M1) ≈
            naive_loss_w_mask(x, l0, M1; lossf = logitcrossentropy, agg = mean)
        @test logitcrossentropy(sum, x, l, M2) ≈
            naive_loss_w_mask(x, l, M2; lossf = logitcrossentropy, agg = sum)
        @test logitcrossentropy(mean, x, l, M2) ≈
            naive_loss_w_mask(x, l, M2; lossf = logitcrossentropy, agg = mean)
        @test logitcrossentropy(sum, x, l0, M2) ≈
            naive_loss_w_mask(x, l0, M2; lossf = logitcrossentropy, agg = sum)
        @test logitcrossentropy(mean, x, l0, M2) ≈
            naive_loss_w_mask(x, l0, M2; lossf = logitcrossentropy, agg = mean)

        @test Flux.gradient(x->logitcrossentropy(sum, x, l, M0), x)[1] ≈
            Flux.gradient(x->naive_loss_w_mask(x, l, M0; lossf = logitcrossentropy, agg = sum), x)[1]
        @test Flux.gradient(x->logitcrossentropy(mean, x, l, M0), x)[1] ≈
            Flux.gradient(x->naive_loss_w_mask(x, l, M0; lossf = logitcrossentropy, agg = mean), x)[1]
        @test Flux.gradient(x->logitcrossentropy(sum, x, l0, M0), x)[1] ≈
            Flux.gradient(x->naive_loss_w_mask(x, l0, M0; lossf = logitcrossentropy, agg = sum), x)[1]
        @test Flux.gradient(x->logitcrossentropy(mean, x, l0, M0), x)[1] ≈
            Flux.gradient(x->naive_loss_w_mask(x, l0, M0; lossf = logitcrossentropy, agg = mean), x)[1]
        @test Flux.gradient(x->logitcrossentropy(sum, x, l, M1), x)[1] ≈
            Flux.gradient(x->naive_loss_w_mask(x, l, M1; lossf = logitcrossentropy, agg = sum), x)[1]
        @test Flux.gradient(x->logitcrossentropy(mean, x, l, M1), x)[1] ≈
            Flux.gradient(x->naive_loss_w_mask(x, l, M1; lossf = logitcrossentropy, agg = mean), x)[1]
        @test Flux.gradient(x->logitcrossentropy(sum, x, l0, M1), x)[1] ≈
            Flux.gradient(x->naive_loss_w_mask(x, l0, M1; lossf = logitcrossentropy, agg = sum), x)[1]
        @test Flux.gradient(x->logitcrossentropy(mean, x, l0, M1), x)[1] ≈
            Flux.gradient(x->naive_loss_w_mask(x, l0, M1; lossf = logitcrossentropy, agg = mean), x)[1]
        @test Flux.gradient(x->logitcrossentropy(sum, x, l, M2), x)[1] ≈
            Flux.gradient(x->naive_loss_w_mask(x, l, M2; lossf = logitcrossentropy, agg = sum), x)[1]
        @test Flux.gradient(x->logitcrossentropy(mean, x, l, M2), x)[1] ≈
            Flux.gradient(x->naive_loss_w_mask(x, l, M2; lossf = logitcrossentropy, agg = mean), x)[1]
        @test Flux.gradient(x->logitcrossentropy(sum, x, l0, M2), x)[1] ≈
            Flux.gradient(x->naive_loss_w_mask(x, l0, M2; lossf = logitcrossentropy, agg = sum), x)[1]
        @test Flux.gradient(x->logitcrossentropy(mean, x, l0, M2), x)[1] ≈
            Flux.gradient(x->naive_loss_w_mask(x, l0, M2; lossf = logitcrossentropy, agg = mean), x)[1]
    end
end
