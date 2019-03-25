
# Table of Contents

1.  [Transformers.jl](#org55bbbed)
2.  [implemented model](#orge06cab7)
3.  [Issue](#org4c758f8)
4.  [Roadmap](#org3e70e94)


<a id="org55bbbed"></a>

# Transformers.jl

Julia implementation of NLP models, that based on google [transformer](https://arxiv.org/abs/1706.03762), with [Flux.jl](https://github.com/FluxML/Flux.jl).
For using the model, see `example` folder.

Install:

    ]add Transformers#master


<a id="orge06cab7"></a>

# implemented model

-   [Attention is all you need](https://arxiv.org/abs/1706.03762)
-   [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)


<a id="org4c758f8"></a>

# Issue

Currently the code is really ugly, need refactor, test and docs.


<a id="org3e70e94"></a>

# Roadmap

-   write docs
-   write test
-   refactor code
-   optimize performance
-   text related util functions
-   better dataset API
-   more datasets
-   openai gpt model
-   openai gpt-2 model
-   google bert model

