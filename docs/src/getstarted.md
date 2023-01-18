# Get Started

Transformers.jl contain multiple functionalities, from basic building block for a transformer model to using pretrained
 model from huggingface. Each of them is put under different submodule in Transformers.jl.


You should find more examples in the [example folder](https://github.com/chengchingwen/Transformers.jl/tree/master/example).

## Create a N-layered Transformer model

All model building block are in the `Layers` module. Here we create a simple 3 layered vanilla transformer
 (multi-head self-attention + MLP) model, where each attention have 4 heads:

```julia
using Flux
using Transformers

num_layer = 3
hidden_size = 128
num_head = 4
head_hidden_size = div(hidden_size, num_head)
intermediate_size = 4hidden_size

trf_blocks = Transformer(Layers.TransformerBlock,
    num_layer, relu, num_head, hidden_size, head_hidden_size, intermediate_size)
```

The `Transformer` layer, by default take a `NamedTuple` as input, and always return a `NamedTuple`.

```julia-repl
julia> x = randn(Float32, hidden_size, 10 #= sequence length =#, 2 #= batch size =#);

julia> y = trf_blocks(
    (; hidden_state = x) ); # The model would be apply on the `hidden_state` field.

julia> keys(y)
(:hidden_state,)
```

It also works on high dimension input data:

```julia-repl
julia> x2 = reshape(x, hidden_size, 5, 2 #= width 5 x heigh 2 =#, 2 #= batch size =#);

julia> y2 = trf_blocks( (; hidden_state = x2) );

julia> y.hidden_state ≈ reshape(y2.hidden_state, size(y.hidden_state))
true
```

Some times you might want to see how the attention score looks like, this can be done by creating a model that return
 the attention score as well. The attention score would usually be in shape (key length, query length, head,
 batch size):

```julia
# creating new model
trf_blocks_ws = Transformer(Layer.TransformerBlock,
    num_layer, relu, num_head, hidden_size, head_hidden_size, intermediate_size;
    return_score = true)

# or transform an old model
trf_blocks_ws = Layers.WithScore(trf_blocks)
```

```julia-repl
julia> y = trf_blocks_ws( (; hidden_state = x) );

julia> keys(y)
(:hidden_state, :attention_score)

```

The model can also take an attention mask to avoid attention looking at the padding tokens. The attention mask would need
 construct with NeuralAttentionlib.Masks:

```julia-repl
julia> mask = Masks.LengthMask([5, 7]); # specify the sequence length of each sample in the batch

julia> y3 = trf_blocks_ws( (; hidden_state = x, attention_mask = mask) );

julia> keys(y3)
(:hidden_state, :attention_mask, :attention_score)

```


## Create a N-layered Transformer Decoder model

For constructing the transformer decoder in the encoder-decoder architecture:

```julia
trf_dec_blocks_ws = Transformer(Layers.TransformerDecoderBlock,
    num_layer, relu, num_head, hidden_size, head_hidden_size, intermediate_size;
	return_score = true)
```

```julia-repl
julia> x3 = x = randn(Float32, hidden_size, 7 #= sequence length =#, 2 #= batch size =#);

julia> z = trf_dec_blocks_ws( (; hidden_state = x3, memory = y.hidden_state #= encoder output =#) );

julia> keys(z)
(:hidden_state, :memory, :cross_attention_score)

julia> size(z.cross_attention_score) # (key length, query length, head, batch size)
(10, 7, 4, 2)

```


## Preprocessing Text

Text processing functionalities are in the `TextEncoders` module.


## Using (HuggingFace) Pre-trained Models

Use the `HuggingFace` module for loading the pretrained model.
