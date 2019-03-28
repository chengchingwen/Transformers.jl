
# Table of Contents

1.  [Transformers.jl](#org038119e)
2.  [implemented model](#org2bcbf10)
3.  [Issue](#orga2755b7)
4.  [Roadmap](#org1940678)


<a id="org038119e"></a>

# Transformers.jl

Julia implementation of NLP models, that based on google [transformer](https://arxiv.org/abs/1706.03762), with [Flux.jl](https://github.com/FluxML/Flux.jl).
For using the model, see `example` folder.

Install:

    ]add Transformers
    
    #Currently the Dataset need the HTTP#master to download
    ]add HTTP#master


<a id="org2bcbf10"></a>

# implemented model

-   [Attention is all you need](https://arxiv.org/abs/1706.03762)
-   [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)


<a id="orga2755b7"></a>

# Issue

Currently the code is really ugly, need refactor, test and docs.


<a id="org1940678"></a>

# Roadmap

-   write docs
-   write test
-   refactor code
-   remove calling `gpu` directly in internal codes
-   better embedding functions
-   lazy CuArrays loading
-   using HTTP to handle dataset download (need HTTP.jl update)
-   optimize performance
-   text related util functions
-   better dataset API
-   more datasets
-   openai gpt model
-   openai gpt-2 model
-   google bert model

