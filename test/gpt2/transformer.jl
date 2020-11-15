@testset "Transformer" begin
  gpt = Gpt2(300, 10, 500, 3)

  x = randn(Float32, 300, 5, 2)
  x1 = x[:,:,1]

  y = gpt(x)
  y1 = gpt(x1)

  @test size(y) == (300, 5, 2)
  @test size(y1) == (300, 5)
  @test y[:,:,1] ≈ y1

  _, yall = gpt(x; return_all_outputs=true)
  @test gpt.blocks.models[1](x) ≈ yall[1]
  @test gpt.blocks.models[2](yall[1]) ≈ yall[2]
  @test gpt.blocks.models[3](yall[2]) ≈ yall[3]

  _, yall1 = gpt(x1; return_all_outputs=true)
  @test gpt.blocks.models[1](x1) ≈ yall1[1]
  @test gpt.blocks.models[2](yall1[1]) ≈ yall1[2]
  @test gpt.blocks.models[3](yall1[2]) ≈ yall1[3]


  mask = cat(reshape([1,1,1,1,1], 1, 5, 1),
             reshape([1,1,1,0,0], 1, 5, 1); dims=3)
  amask = getmask(mask, mask)

  ym, ymall = gpt(x, mask; return_all_outputs=true)
  topo = @nntopo ((x, m) => x:(x, m)) => 3
  @test topo(gpt.blocks.models, x, amask) .* mask ≈ ym
end
