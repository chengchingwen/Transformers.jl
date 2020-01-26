@testset "Gpt2" begin
  using Transformers.GenerativePreTrain2
  for f ∈ readdir("./gpt2/")
    include("./gpt2/$f")
  end
end
