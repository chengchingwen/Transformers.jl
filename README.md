# Transformers.jl

[![Build Status](https://travis-ci.com/chengchingwen/Transformers.jl.svg?branch=master)](https://travis-ci.com/chengchingwen/Transformers.jl)
[![codecov](https://codecov.io/gh/chengchingwen/Transformers.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chengchingwen/Transformers.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://chengchingwen.github.io/Transformers.jl/dev/)

Julia implementation of NLP models, that based on google [transformer](https://arxiv.org/abs/1706.03762), with [Flux.jl](https://github.com/FluxML/Flux.jl).
For using the model, see `example` folder.


# Installation

In the Julia REPL:

    ]add Transformers
    
For using GPU, install & build:

    ]add CuArrays
    
    ]build 
    
    julia> using CuArrays
    
    julia> using Transformers
    
    #run the model below
    .
    .
    .


# Implemented model
You can find the code in `example` folder.

-   [Attention is all you need](https://arxiv.org/abs/1706.03762)
-   [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
-   [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

# Example
Take a simple encoder-decoder model construction of machine translation task. With `Transformers.jl` we can easily define/stack the models. 

```julia
using Transformers
using Transformers.Basic

encoder = Stack(
    @nntopo(e → pe:(e, pe) → x → x → $N),
    PositionEmbedding(512),
    (e, pe) -> e .+ pe,
    Dropout(0.1),
    [Transformer(512, 8, 64, 2048) for i = 1:N]...
)

decoder = Stack(
    @nntopo((e, m, mask):e → pe:(e, pe) → t → (t:(t, m, mask) → t:(t, m, mask)) → $N:t → c),
    PositionEmbedding(512),
    (e, pe) -> e .+ pe,
    Dropout(0.1),
    [TransformerDecoder(512, 8, 64, 2048) for i = 1:N]...,
    Positionwise(Dense(512, length(labels)), logsoftmax)
)

function loss(src, trg, src_mask, trg_mask)
    label = onehot(vocab, trg)

    src = embedding(src)
    trg = embedding(trg)

    mask = getmask(src_mask, trg_mask)

    enc = encoder(src)
    dec = decoder(trg, enc, mask)

    loss = logkldivergence(label, dec[:, 1:end-1, :], trg_mask[:, 1:end-1, :])
end
```

See `example` folder for the complete example.

# Roadmap

## What we have in v0.1.0

-   `Transformer` and `TransformerDecoder` support for both 2d & 3d data.
-   `PositionEmbedding` implementation.
-   `Positionwise` for handling 2d & 3d input.
-   docstring for most of the functions.
-   runable examples (see `example` folder)


## What we will have in v0.2.0

-   The BERT model (since it's part of the JSoC 2019)
-   tutorials
-   complete GPT APIs
-   GPT-2 model
-   docs site for this project
-   benchmarks
-   more examples


## What we might have in v0.2.0 (If we are lucky)
-   TPU support with XLA.jl
-   complete docs for datasets
-   more datasets support


## Messy checklist

-   <code>[33%]</code> write docs
    -   [X] docstring
    -   [ ] examples
    -   [ ] make docs site
-   [X] write test
-   [ ] refactor code
-   [ ] optimize performance
-   [ ] better dataset API
-   [ ] more datasets
-   <code>[75%]</code> openai gpt model
    -   [X] model implementation
    -   [X] loading pretrain
    -   [X] model example
    -   [ ] more util functions
-   [ ] openai gpt-2 model
-   [ ] google bert model
-   [ ] TPU support
-   [ ] openai sparse transformer
-   [ ] benchmarks


