# Transformers.jl

Julia implementation of NLP models, that based on google [transformer](https://arxiv.org/abs/1706.03762), with [Flux.jl](https://github.com/FluxML/Flux.jl).
For using the model, see `example` folder.


# Table of Contents

1.  [Installation](#orgcf1a8ed)
2.  [implemented model](#orgd7f4e33)
3.  [Issue](#org462b271)
4.  [Roadmap](#org4c326d7)


<a id="orgcf1a8ed"></a>

# Installation

In the Julia REPL:

    ]add Transformers
    
    #Currently the Dataset need the HTTP#master to download
    ]add HTTP#master

For using GPU, install & build:

    ]add CuArrays
    
    ]build 
    
    julia> using CuArrays
    
    julia> using Transformers
    
    #run the model below
    .
    .
    .


<a id="orgd7f4e33"></a>

# implemented model

-   [Attention is all you need](https://arxiv.org/abs/1706.03762)
-   [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)


<a id="org462b271"></a>

# Issue

Currently the code is really ugly, need refactor, test and docs.


<a id="org4c326d7"></a>

# Roadmap

-   [ ] write docs
-   [ ] write test
-   [ ] refactor code
-   <code>[60%]</code> better embedding functions
    -   [X] gather function forward
    -   [X] gather function backward (might be better)
    -   [X] OneHotArray
    -   [ ] more util functions
    -   [ ] easy gpu data
-   [X] lazy CuArrays loading
-   [ ] using HTTP to handle dataset download (need HTTP.jl update)
-   [ ] optimize performance
-   [ ] text related util functions
-   [ ] better dataset API
-   [ ] more datasets
-   <code>[75%]</code> openai gpt model
    -   [X] model implementation
    -   [X] loading pretrain
    -   [X] model example
    -   [ ] more util functions
-   [ ] openai gpt-2 model
-   [ ] google bert model

