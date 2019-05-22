# Transformers.jl

[![Build Status](https://travis-ci.com/chengchingwen/Transformers.jl.svg?branch=master)](https://travis-ci.com/chengchingwen/Transformers.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/ns0x37a7sykjuhw0?svg=true)](https://ci.appveyor.com/project/chengchingwen/transformers-jl)
[![codecov](https://codecov.io/gh/chengchingwen/Transformers.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chengchingwen/Transformers.jl)

Julia implementation of NLP models, that based on google [transformer](https://arxiv.org/abs/1706.03762), with [Flux.jl](https://github.com/FluxML/Flux.jl).
For using the model, see `example` folder.


# Installation

In the Julia REPL:

    ]add Transformers
    
    #Currently the Dataset need the HTTP#master to download WMT
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


# Implemented model
You can find the code in `example` folder.

-   [Attention is all you need](https://arxiv.org/abs/1706.03762)
-   [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)


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


# Usage

## Transformer

The `Transformer` and `TransformerDecoder` is the encoder and decoder block of the origin paper, and they are all implement as the 
regular Flux Layer, so you can treat them just as the `Dense` layer. See the docstring for the argument. However, for the sequence 
data input, we usually have a 3 dimensional input of shape `(hidden size, sequence length, batch size)` instead of just `(hidden size, batch size)`. 
Therefore, we implement both 2d & 3d operation according to the input type (The `N` of `Array{T, N}`). We are able to handle both input of shape 
`(hidden size, sequence length, batch size)` and `(hidden size, sequence length)` for the case with only 1 input.

```julia
using Transfomers

m = Transformer(512, 8, 64, 2048) #define a Transformer block with 8 head and 64 neuron for each head
x = randn(512, 30, 3) #fake data of length 30

y = m(x)
```


## Positionwise

For the sequential task, we need to handle the 3 dimensional input. However, most of the layer in Flux only support input with shape 
`(hidden size, batch size)`. In order to tackle this problem, we implement the `Positionwise` helper function that is almost the same 
as `Flux.Chain` but it will run the model position-wisely. (internally it just reshape the input to 2d and apply the model then reshape 
back). 

```julia
using Transformers
using Flux

m = Positionwise(Dense(10, 5), Dense(5, 2), softmax)
x = randn(10, 30, 3)

y = m(x)

# which is equivalent to 
# 
# m = Chain(Dense(10, 5), Dense(5, 2), softmax)
# x1 = randn(10, 30)
# x2 = randn(10, 30)
# x3 = randn(10, 30)
# y = cat(m(x1), m(x2), m(x3); dims=3)
```


## PositionEmbedding

We implement two kinds of position embedding, one is based on the sin/cos function (mentioned in the paper, 
attention is all you need). Another one is just like regular word embedding but with the position index. The 
first argument is the `size`. Since the position embedding is only related to the length of the input (
we use `size(input, 2)` as the length), the return value of the layer will be the embedding of the given 
length without duplicate to the batch size. you can/should use broadcast add to get the desired output.

```julia
# sin/cos based position embedding which is not trainable
pe = PositionEmbedding(10) # or PositionEmbedding(10; trainable = false)

# trainable position embedding need to specify the maximum length
pe = PositionEmbedding(10, 1024; trainable = true)

x = randn(Float32, 10, 6, 3) #fake data of shape (10, length = 6, batched_size = 3)

e = pe(x) #get the position embedding
y = x .+ e # add the position embedding to each sample
```


## The Stack NNTopo DSL

Since the `TransformerDecoder` require more than one input, it's not convenient to use with `Chain`. Therefore, we implement a very simple 
DSL(Domain Specific Language) to handle the function structure. You can use the `@nntopo` macro to define the structure then call the function 
with the given model.

### NNTopo Syntax

we call the DSL NNTopo for "Neural Network Topology", but actually it is just used to define where the input & output should be in a sequence of 
function, or the complex version of the `|>` function in Julia.

1.  "Chain" the functions

For example:

```julia
y = h(f(g(x))) #a chain of function call

# or 
a = g(x)
b = f(a)
y = h(b)

# is equivalent to 
topo = @nntopo x => a => b => y # first we define the topology/architecture
y = topo((g, f, h), x) #then call on the given functions
```

each `=>` is a function call, left hand side is the input argument and right hand side is the output name.


2.  Loop unrolling

you can also unroll a loop:

```julia
y = g(f(f(f(f(x)))))

# or 
tmp = x
for i = 1:4
tmp = f(tmp)
end
y = g(tmp)

# is equivalent to 
topo = @nntopo x => 4 => y
y = topo((f,f,f,f, g), x) # f can also be different
```

3.  Multiple argument & jump connection

As we metioned above, the original intention was to handle the case that we have more than one input & output. So, we can do this with the following syntax: 

```julia
# a complex structure
# x1 to x4 in the given inputs
t = f(x1, x2)
z1, z2 = g(t, x3)
w = h(x4, z1)
y = k(x2, z2, w)

# is equivalent to 
topo = @nntopo (x1, x2, x3, x4):(x1, x2) => t:(t, x3) => (z1, z2):(x4, z1) => w:(x2, z2, w) => y
y = topo((f, g, h, k), x1, x2, x3, x4)

# you can also see the function with `print_topo` function
using Transformers.Basic: print_topo

print_topo(topo; models=(f, g, h, k))
# 
# NNTopo{"(x1, x2, x3, x4):(x1, x2) => (t:(t, x3) => ((z1, z2):(x4, z1) => (w:(x2, z2, w) => y)))"}
# topo_func(model, x1, x2, x3, x4)
#         t = f(x1, x2)
#         (z1, z2) = g(t, x3)
#         w = h(x4, z1)
#         y = k(x2, z2, w)
#         y
# end
```

4.  Specify the variables you want

Notice that we use a `:` to seperate the input/output variables name for each function call, if the `:` is not present, we will by default assume 
the output variables are all the inputs of the next function call. i.e. `x => (t1, t2) => y` is equal to `x => (t1, t2):(t1, t2) => y`. 

We can also return multiple variables, so the complete syntax can be viewed as:
    
        (input arguments):(function1 inputs) => (function1 outputs):(function2 inputs):(function2 outputs) => .... => (function_n outputs):(return variables)

5.  Interpolation

we also support interpolation, so you can use a variable to hold a substructure or the unroll number. But **notice** that the 
interpolation variable should always be at the top level of the module since we can only get that value with `eval`. To use 
interpolte local variables, use `@nntopo_str "topo_pattern"` instead.

```julia
N = 3
topo = @nntopo((e, m, mask):e → pe:(e, pe) → t → (t:(t, m, mask) → t:(t, m, mask)) → $N:t → c)

# or
# topo = @nntopo_str "(e, m, mask):e → pe:(e, pe) → t → (t:(t, m, mask) → t:(t, m, mask)) → $N:t → c"

print_topo(topo)
# 
# NNTopo{"(e, m, mask):e → (pe:(e, pe) → (t → ((t:(t, m, mask) → t:(t, m, mask)) → (3:t → c))))"}
# topo_func(model, e, m, mask)
#         pe = model[1](e)
#         t = model[2](e, pe)
#         t = model[3](t)
#         t = model[4](t, m, mask)
#         t = model[5](t, m, mask)
#         t = model[6](t, m, mask)
#         c = model[7](t)
#         c
# end
```

6.  Nested Structure

you can also use the `()` to create a nested structure for the unroll.

```julia
topo = @nntopo x => ((y => z => t) => 3 => w) => 2
print_topo(topo)
# 
# NNTopo{"x => (((y => (z => t)) => (3 => w)) => 2)"}
# topo_func(model, x)
#         y = model[1](x)
#         z = model[2](y)
#         t = model[3](z)
#         z = model[4](t)
#         t = model[5](z)
#         z = model[6](t)
#         t = model[7](z)
#         w = model[8](t)
#         z = model[9](w)
#         t = model[10](z)
#         z = model[11](t)
#         t = model[12](z)
#         z = model[13](t)
#         t = model[14](z)
#         w = model[15](t)
#         w
# end
```

7. Collect Variables

you can also collect some variables that you are interested in with `'` on that variable. For example:

```julia
julia> @nntopo x => y' => 3 => z
NNTopo{"x => (y' => (3 => z))"}
topo_func(model, x)
        y = model[1](x)
        %1 = y
        y = model[2](y)
        %2 = y
        y = model[3](y)
        %3 = y
        y = model[4](y)
        %4 = y
        z = model[5](y)
        (z, (%1, %2, %3, %4))
end

julia> @nntopo (x,y) => (a,b,c,d') => (w',r',y) => (m,n)' => z
NNTopo{"(x, y) => ((a, b, c, d') => ((w', r', y) => (((m, n))' => z)))"}
topo_func(model, x, y)
        (a, b, c, d) = model[1](x, y)
        %1 = d
        (w, r, y) = model[2](a, b, c, d)
        %2 = (w, r)
        (m, n) = model[3](w, r, y)
        %3 = (m, n)
        z = model[4](m, n)
        (z, (%1, %2, %3))
end
```

### Stack

With the NNTopo DSL, now we can simple use the NNTopo with our Stack type, which is also like the `Chain` but we also need to pass in the 
`topo` for the architecture. You can check the actual function call with `show_stackfunc`.

```julia
#The Decoder Example in Attention is All you need
using Transformers.Stacks
Stack(
@nntopo((e, m, mask):e → pe:(e, pe) → t → (t:(t, m, mask) → t:(t, m, mask)) → $N:t → c),
PositionEmbedding(512),
(e, pe) -> e .+ pe,
Dropout(0.1),
[TransformerDecoder(512, 8, 64, 2048) for i = 1:N]...,
Positionwise(Dense(512, length(labels)), logsoftmax)
)

julia> show_stackfunc(s)
topo_func(model, e, m, mask)
        pe = PositionEmbedding(512)(e)
        t = getfield(Main, Symbol("##23#25"))()(e, pe)
        t = Dropout{Float64}(0.1, true)(t)
        t = TransformerDecoder(head=8, head_size=64, pwffn_size=2048, size=512, dropout=0.1)(t, m, mask)
        t = TransformerDecoder(head=8, head_size=64, pwffn_size=2048, size=512, dropout=0.1)(t, m, mask)
        t = TransformerDecoder(head=8, head_size=64, pwffn_size=2048, size=512, dropout=0.1)(t, m, mask)
        c = Positionwise{Tuple{Dense{typeof(identity),TrackedArray{…,Array{Float32,2}},TrackedArray{…,Array{Float32,1}}},typeof(logsoftmax)}}((Dense(512, 12), NNlib.logsoftmax))(t)
        c
end
```


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
-   <code>[83%]</code> better embedding functions
    -   [X] gather function forward
    -   [X] gather function backward (might be better)
    -   [X] OneHotArray
    -   [ ] more util functions
    -   [X] easy gpu data
    -   [X] remove Vocabulary
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
-   [ ] TPU support
-   [ ] openai sparse transformer
-   [ ] benchmarks


