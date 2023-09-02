<div align="center"> <img src="images/transformerslogo.png" alt="Transformers.jl" width="512"></img></div>

[![Build status](https://github.com/chengchingwen/Transformers.jl/workflows/CI/badge.svg)](https://github.com/chengchingwen/Transformers.jl/actions)
[![codecov](https://codecov.io/gh/chengchingwen/Transformers.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chengchingwen/Transformers.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://chengchingwen.github.io/Transformers.jl/dev/)

Julia implementation of [transformer](https://arxiv.org/abs/1706.03762)-based models, with [Flux.jl](https://github.com/FluxML/Flux.jl).

*notice: The current version is almost completely different from the 0.1.x version. If you are using the old version, make sure to update the changes or stick to the old version.*

# Installation

In the Julia REPL:

    ]add Transformers


# Example

Using pretrained Bert with `Transformers.jl`.

```julia
using Transformers
using Transformers.TextEncoders
using Transformers.HuggingFace

textencoder, bert_model = hgf"bert-base-uncased"

text1 = "Peter Piper picked a peck of pickled peppers"
text2 = "Fuzzy Wuzzy was a bear"

text = [[ text1, text2 ]] # 1 batch of contiguous sentences
sample = encode(textencoder, text) # tokenize + pre-process (add special tokens + truncate / padding + one-hot encode)

@assert reshape(decode(textencoder, sample.token), :) == [
    "[CLS]", "peter", "piper", "picked", "a", "peck", "of", "pick", "##led", "peppers", "[SEP]",
    "fuzzy", "wu", "##zzy",  "was", "a", "bear", "[SEP]"
]

bert_features = bert_model(sample).hidden_state
```

See `example` folder for the complete example.


# For more information

If you want to know more about this package, see the [document](https://chengchingwen.github.io/Transformers.jl/dev/)
 and read code in the `example` folder. You can also tag me (@chengchingwen) on Julia's slack or discourse if
 you have any questions, or just create a new Issue on GitHub.
