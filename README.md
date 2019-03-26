
# Table of Contents

1.  [Transformers.jl](#orgb2bedeb)
2.  [implemented model](#org51e63fb)
3.  [Issue](#orgf87b327)
4.  [Roadmap](#org201ee68)


<a id="orgb2bedeb"></a>

# Transformers.jl

Julia implementation of NLP models, that based on google [transformer](https://arxiv.org/abs/1706.03762), with [Flux.jl](https://github.com/FluxML/Flux.jl).
For using the model, see `example` folder.

Install:

    ]add Transformers#master
    
    #Currently the Dataset need the HTTP#master to download
    ]add HTTP#master


<a id="org51e63fb"></a>

# implemented model

-   [Attention is all you need](https://arxiv.org/abs/1706.03762)
-   [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)


<a id="orgf87b327"></a>

# Issue

Currently the code is really ugly, need refactor, test and docs.


<a id="org201ee68"></a>

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

