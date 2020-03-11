@testset "Transformer" begin
  bert = Bert(300, 10, 500, 3)

  x = randn(Float32, 300, 5, 2)
  x1 = x[:,:,1]

  y = bert(x)
  y1 = bert(x1)

  @test size(y) == (300, 5, 2)
  @test size(y1) == (300, 5)
  @test y[:,:,1] ≈ y1

  _, yall = bert(x; all=true)
  @test bert.ts.models[1](x) ≈ yall[1]
  @test bert.ts.models[2](yall[1]) ≈ yall[2]
  @test bert.ts.models[3](yall[2]) ≈ yall[3]

  _, yall1 = bert(x1; all=true)
  @test bert.ts.models[1](x1) ≈ yall1[1]
  @test bert.ts.models[2](yall1[1]) ≈ yall1[2]
  @test bert.ts.models[3](yall1[2]) ≈ yall1[3]


  mask = cat(reshape([1,1,1,1,1], 1, 5, 1),
             reshape([1,1,1,0,0], 1, 5, 1); dims=3)
  amask = getmask(mask, mask)

  ym, ymall = bert(x, mask; all=true)
  topo = @nntopo ((x, m) => x:(x, m)) => 3
  @test topo(bert.ts.models, x, amask) .* mask ≈ ym
end
