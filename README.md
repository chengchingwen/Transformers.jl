
# Table of Contents

1.  [Transformers.jl](#org65cf6ea)
2.  [implemented model](#org417a5ad)
3.  [Issue](#orgb6d8d3f)
4.  [Roadmap](#org63f3074)


<a id="org65cf6ea"></a>

# Transformers.jl

Julia implementation of NLP models, that based on google [transformer](https://arxiv.org/abs/1706.03762), with [Flux.jl](https://github.com/FluxML/Flux.jl).
For using the model, see `example` folder.

Install:

    ]add Transformers
    
    #Currently the Dataset need the HTTP#master to download
    ]add HTTP#master


<a id="org417a5ad"></a>

# implemented model

-   [Attention is all you need](https://arxiv.org/abs/1706.03762)
-   [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)


<a id="orgb6d8d3f"></a>

# Issue

Currently the code is really ugly, need refactor, test and docs.


<a id="org63f3074"></a>

# Roadmap

-   [ ] write docs
-   [ ] write test
-   [ ] refactor code
-   [X] better embedding functions
-   [ ] lazy CuArrays loading
-   [ ] using HTTP to handle dataset download (need HTTP.jl update)
-   [ ] optimize performance
-   [ ] text related util functions
-   [ ] better dataset API
-   [ ] more datasets
-   [X] openai gpt model
-   [ ] openai gpt-2 model
-   [ ] google bert model

