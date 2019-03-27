
# Table of Contents

1.  [Transformers.jl](#org5853523)
2.  [implemented model](#org8c7be0a)
3.  [Issue](#orga4198ca)
4.  [Roadmap](#org9151d19)


<a id="org5853523"></a>

# Transformers.jl

Julia implementation of NLP models, that based on google [transformer](https://arxiv.org/abs/1706.03762), with [Flux.jl](https://github.com/FluxML/Flux.jl).
For using the model, see `example` folder.

Install:

    ]add Transformers
    
    #Currently the Dataset need the HTTP#master to download
    ]add HTTP#master


<a id="org8c7be0a"></a>

# implemented model

-   [Attention is all you need](https://arxiv.org/abs/1706.03762)
-   [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)


<a id="orga4198ca"></a>

# Issue

Currently the code is really ugly, need refactor, test and docs.


<a id="org9151d19"></a>

# Roadmap

-   write docs
-   write test
-   refactor code
-   better embedding functions
-   lazy CuArrays loading
-   optimize performance
-   text related util functions
-   better dataset API
-   more datasets
-   openai gpt model
-   openai gpt-2 model
-   google bert model

