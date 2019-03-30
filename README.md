
# Table of Contents

1.  [Transformers.jl](#org7f099b9)
2.  [implemented model](#orgdf7e5fd)
3.  [Issue](#orgd6197fd)
4.  [Roadmap](#orgc0dc093)


<a id="org7f099b9"></a>

# Transformers.jl

Julia implementation of NLP models, that based on google [transformer](https://arxiv.org/abs/1706.03762), with [Flux.jl](https://github.com/FluxML/Flux.jl).
For using the model, see `example` folder.

Install:

    ]add Transformers
    
    #Currently the Dataset need the HTTP#master to download
    ]add HTTP#master


<a id="orgdf7e5fd"></a>

# implemented model

-   [Attention is all you need](https://arxiv.org/abs/1706.03762)
-   [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)


<a id="orgd6197fd"></a>

# Issue

Currently the code is really ugly, need refactor, test and docs. The grad of gather function is very slow on large array. need better implementation.


<a id="orgc0dc093"></a>

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
-   [ ] openai gpt model
-   [ ] openai gpt-2 model
-   [ ] google bert model

