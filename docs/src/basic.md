# Transformers.Basic
Basic functionality of Transformers.jl, provide the Transformer encoder/decoder implementation and other convenient function.

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

## API Reference

```@autodocs
Modules=[Transformers.Basic]
Order = [:type, :function, :macro]
```
