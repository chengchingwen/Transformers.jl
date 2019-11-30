@testset "Gather" begin
    w = randn(Float32, 10,10)
    wh = randn(Float32, 10,5,4,3)

    cuw = cu(w)
    cuwh = cu(wh)

    ind = rand(1:10, 3,5)

    @test (gather(cuw, todevice([3,5,7])) |> collect) == hcat(map(i->w[:, i], [3,5,7])...)
    @test (gather(cuw, todevice(ind)) |> collect) ==
        cat(map(j-> hcat(map(i->w[:, i], ind[:,j])...), 1:5)...; dims=3)
    @test (gather(cuwh, todevice([(5,3,3) (2,1,2); (5,4,1) (4,2,1)])) |> collect) == begin
        a = wh[:, 5, 3, 3]
        b = wh[:, 2, 1, 2]
        c = wh[:, 5, 4, 1]
        d = wh[:, 4, 2, 1]
        A = hcat(a,c)
        B = hcat(b,d)
        Z = cat(A, B; dims=3)
    end


    ca = cu(randn(5,  30))
    cb = todevice(OneHotArray(30, ones(Int, 20)))

    fa = zeros(Float32, size(ca))
    fa[:, 1] .= 20
    pca_grad = gradient(ca) do pca
        z = pca * cb
      sum(z)
    end
    @test collect(pca_grad[1]) == fa
end
