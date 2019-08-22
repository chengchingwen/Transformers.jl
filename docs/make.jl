using Documenter, Transformers

using Transformers.Basic
using Transformers.Basic: MultiheadAttention
using Transformers.Pretrain
using Transformers.Stacks
using Transformers.GenerativePreTrain
using Transformers.BidirectionalEncoder

makedocs(sitename="Transformers.jl",
         pages = Any[
           "Home" => "index.md",
           "Tutorial" => "tutorial.md",
           "Basic" => "basic.md",
           "Stacks" => "stacks.md",
           "Pretrain" => "pretrain.md",
           "Models" => [
             "GPT" => "gpt.md",
             "BERT" => "bert.md",
           ],
           "Datasets" => "datasets.md"
         ],
         )

deploydocs(
    repo = "github.com/chengchingwen/Transformers.jl.git",
)
