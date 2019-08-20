@testset "Transformer" begin
  bert = Bert(300, 10, 500, 3)
  Flux.testmode!(bert)

  x = randn(300, 5, 2)
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
end
