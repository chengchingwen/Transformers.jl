@testset "Gather" begin
    w = randn(10,10)
    wh = randn(10,5,4,3)

    ind = rand(1:10, 3,5)

    @test gather(w, [3,5,7]) == hcat(map(i->w[:, i], [3,5,7])...)
    @test gather(w, ind) == cat(map(j-> hcat(map(i->w[:, i], ind[:,j])...), 1:5)...; dims=3)
    @test gather(wh, [(5,3,3) (2,1,2); (5,4,1) (4,2,1)]) == begin
        a = wh[:, 5, 3, 3]
        b = wh[:, 2, 1, 2]
        c = wh[:, 5, 4, 1]
        d = wh[:, 4, 2, 1]
        A = hcat(a,c)
        B = hcat(b,d)
        Z = cat(A, B; dims=3)
    end


    ca = randn(512,  40000)
    cb = OneHotArray(40000, ones(Int, 20))

    using Flux: back!, param
    pca = param(ca)

    z = pca * cb
    back!(sum(z))
    fa = zeros(Float32, size(ca))
    fa[:, 1] .= 20
    @test pca.grad == fa

end
