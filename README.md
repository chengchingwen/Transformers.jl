
# Table of Contents

1.  [Transformers.jl](#org24a62fa)
2.  [implemented model](#org8446635)
3.  [implementation detail](#orgcf2de53)
4.  [Issue](#org807c82d)
5.  [Roadmap](#org8a12ea0)


<a id="org24a62fa"></a>

# Transformers.jl

Julia implementation of NLP models, that based on google [transformer](https://arxiv.org/abs/1706.03762), with [Flux.jl](https://github.com/FluxML/Flux.jl).
For using the model, see `example` folder.


<a id="org8446635"></a>

# implemented model

-   [Attention is all you need](https://arxiv.org/abs/1706.03762)
-   [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)


<a id="orgcf2de53"></a>

# implementation detail

There are some hack in the implementation, will be remove once packages update.

-   some hack to make gpu work
-   `batchedmul`: Currently `Flux.jl` doesn't have a batched matrix multiply function, 
    so I implement one.
    -   `batched_gemm!`: borrow the implemetation from `BatchedRoutines.jl`
-   `gelu`: cpu & gpu version of `gelu`, can be remove when `NNlib.jl` & `CuArrays.jl` has one.


<a id="org807c82d"></a>

# Issue

Currently the code is really ugly, need refactor, test and docs.


<a id="org8a12ea0"></a>

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

