
# Table of Contents

1.  [Transformers.jl](#orgd396d0c)
2.  [implemented model](#orgb99b674)
3.  [Issue](#orgc3ce3a4)
4.  [Roadmap](#orgf8d9e4b)


<a id="orgd396d0c"></a>

# Transformers.jl

Julia implementation of NLP models, that based on google [transformer](https://arxiv.org/abs/1706.03762), with [Flux.jl](https://github.com/FluxML/Flux.jl).
For using the model, see `example` folder.

Install:

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


<a id="orgb99b674"></a>

# implemented model

-   [Attention is all you need](https://arxiv.org/abs/1706.03762)
-   [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)


<a id="orgc3ce3a4"></a>

# Issue

Currently the code is really ugly, need refactor, test and docs. The grad of gather function is very slow on large array. need better implementation.


<a id="orgf8d9e4b"></a>

# Roadmap

-   [ ] write docs
-   [ ] write test
-   [ ] refactor code
-   [X] better embedding functions
-   [X] lazy CuArrays loading
-   [ ] using HTTP to handle dataset download (need HTTP.jl update)
-   [ ] optimize performance
-   [ ] text related util functions
-   [ ] better dataset API
-   [ ] more datasets
-   [ ] openai gpt model
-   [ ] openai gpt-2 model
-   [ ] google bert model

