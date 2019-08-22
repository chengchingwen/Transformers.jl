@testset "Gpt" begin
  using Transformers.GenerativePreTrain
  for f ∈ readdir("./gpt/")
    include("./gpt/$f")
  end
end
