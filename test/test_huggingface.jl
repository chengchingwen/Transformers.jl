@testset "HuggingFace" begin
  using Transformers.HuggingFace
  for f ∈ readdir("./huggingface/")
    include("./huggingface/$f")
  end
end
