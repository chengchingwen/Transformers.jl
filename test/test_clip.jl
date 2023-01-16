@testset "CLIPTextModel" begin
    using Transformers.HuggingFace
    using Transformers
    using TextEncodeBase
    using NeuralAttentionlib
    for f ∈ readdir("./clip/")
      include("./clip/$f")
    end
  end