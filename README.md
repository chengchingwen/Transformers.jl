<div align="center"> <img src="images/transformerslogo.png" alt="Transformers.jl" width="512"></img></div>

[![Build status](https://github.com/chengchingwen/Transformers.jl/workflows/CI/badge.svg)](https://github.com/chengchingwen/Transformers.jl/actions)
[![codecov](https://codecov.io/gh/chengchingwen/Transformers.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chengchingwen/Transformers.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://chengchingwen.github.io/Transformers.jl/dev/)

Julia implementation of [transformer](https://arxiv.org/abs/1706.03762)-based models, with [Flux.jl](https://github.com/FluxML/Flux.jl).

# Installation

In the Julia REPL:

    ]add Transformers
    
For using GPU, install & build:

    ]add CUDA
    
    ]build 
    
    julia> using CUDA
    
    julia> using Transformers
    
    #run the model below
    .
    .
    .


# Example

Using pretrained Bert with `Transformers.jl`.

```julia
using Transformers
using Transformers.Basic
using Transformers.Pretrain

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
vocab = Vocabulary(wordpiece)

text1 = "Peter Piper picked a peck of pickled peppers" |> tokenizer |> wordpiece
text2 = "Fuzzy Wuzzy was a bear" |> tokenizer |> wordpiece

text = ["[CLS]"; text1; "[SEP]"; text2; "[SEP]"]
@assert text == [
    "[CLS]", "peter", "piper", "picked", "a", "peck", "of", "pick", "##led", "peppers", "[SEP]", 
    "fuzzy", "wu", "##zzy",  "was", "a", "bear", "[SEP]"
]

token_indices = vocab(text)
segment_indices = [fill(1, length(text1)+2); fill(2, length(text2)+1)]

sample = (tok = token_indices, segment = segment_indices)

bert_embedding = sample |> bert_model.embed
feature_tensors = bert_embedding |> bert_model.transformers
```

See `example` folder for the complete example.


# Huggingface

We have some support for the models from [`huggingface/transformers`](https://github.com/huggingface/transformers).

```julia
using Transformers.HuggingFace

# loading a model from huggingface model hub
julia> model = hgf"bert-base-cased:forquestionanswering";
┌ Warning: Transformers.HuggingFace.HGFBertForQuestionAnswering doesn't have field cls.
└ @ Transformers.HuggingFace ~/peter/repo/gsoc2020/src/huggingface/models/models.jl:46
┌ Warning: Some fields of Transformers.HuggingFace.HGFBertForQuestionAnswering aren't initialized with loaded state: qa_outputs
└ @ Transformers.HuggingFace ~/peter/repo/gsoc2020/src/huggingface/models/models.jl:52

```

Current we only support a few model and the tokenizer part is not finished yet.


# For more information

If you want to know more about this package, see the [document](https://chengchingwen.github.io/Transformers.jl/dev/) 
and the series of [blog posts](https://nextjournal.com/chengchingwen) I wrote for JSoC and GSoC. You can also 
tag me (@chengchingwen) on Julia's slack or discourse if you have any questions, or just create a new Issue on GitHub.


# Roadmap

## What we have before v0.2

-   `Transformer` and `TransformerDecoder` support for both 2d & 3d data.
-   `PositionEmbedding` implementation.
-   `Positionwise` for handling 2d & 3d input.
-   docstring for most of the functions.
-   runable examples (see `example` folder)
-   `Transformers.HuggingFace` for handling pretrains from `huggingface/transformers`

## What we will have in v0.2.0

-   Complete tokenizer APIs
-   tutorials
-   benchmarks
-   more examples
