# Tutorial

The following content will cover the basic introductions about the Transformer model and the implementation.

## Transformer model

The Transformer model was proposed in the paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762). In that paper they provide a new way of handling the sequence transduction problem (like the machine translation task) without complex recurrent or convolutional structure. Simply use a stack of attention mechanisms to get the latent structure in the input sentences and a special embedding (positional embedding) to get the locationality. The whole model architecture looks like this:

The Transformer model architecture (picture from the origin paper)![transformer](./transformerblocks.png)

### Multi-Head Attention

Instead of using the regular attention mechanism, they split the input vector to several pairs of subvector and perform a dot-product attention on each subvector pairs.  

regular attention v.s. Multi-Head attention (picture from the origin paper)![mhatten](mhatten.png)

For those who like mathematical expression, here is the formula:

```math
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
```

```math
MultiHead(Q, K, V) = Concat(head_1,..., head_h)W^O
\text{where }head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

### Positional Embedding

As we mentioned above, transformer model didn't depend on the recurrent or convolutional structure. On the other hand, we still need a way to differentiate two sequence with same words but different order. Therefore, they add the locational information on the embedding, i.e. the origin word embedding plus a special embedding that indicate the order of that word. The special embedding can be computed by some equations or just use another trainable embedding matrix. In the paper, the positional embedding use this formula:

```math
PE_{(pos, k)} = \begin{cases}
sin(\frac{pos}{10^{4k/d_k}}),& \text{if }k \text{ is even}\\
cos(\frac{pos}{10^{4k/d_k}}), & \text{if }k \text{ is odd}
\end{cases}
```

where $pos$ is the locational information that tells you the given word is the $pos$\-th word, and $k$ is the $k$\-th dimension of the input vector. $d\_k$ is the total length of the word/positional embedding. So the new embedding will be computed as:

```math
Embedding_k(word) = WordEmbedding_k(word) + PE(pos\_of\_word, k)
```

## Transformers.jl

Now we know how the transformer model looks like, let's take a look at the Transformers.jl. The package is build on top of a famous deep learning framework in Julia, [Flux.jl](https://github.com/FluxML/Flux.jl/).

### Example

To best illustrate the usage of Transformers.jl, we will start with building a two layer Transformer model on a sequence copy task. Before we start, we need to install all the package we need:

```julia
using Pkg
Pkg.add("CuArrays")
Pkg.add("Flux")
Pkg.add("Transformers")
```

We use[ CuArrays.jl](https://github.com/JuliaGPU/CuArrays.jl) for the GPU support. 

```julia
using Flux
using CuArrays
using Transformers
using Transformers.Basic #for loading the positional embedding
```

### Copy task

The copy task is a toy test case of a sequence transduction problem that simply return the same sequence as the output. Here we define the input as a random sequence with number from 1~10 and length 10. we will also need a start and end symbol to indicate where is the begin and end of the sequence. We can use `Transformers.Basic.Vocabulary` to turn the input to corresponding index.

```julia
labels = collect(1:10)
startsym = 11
endsym = 12
unksym = 0
labels = [unksym, startsym, endsym, labels...]
vocab = Vocabulary(labels, unksym)
```

```julia
#function for generate training datas
sample_data() = (d = rand(1:10, 10); (d,d))
#function for adding start & end symbol
preprocess(x) = [startsym, x..., endsym]

@show sample = preprocess.(sample_data())
@show encoded_sample = vocab(sample[1]) #use Vocabulary to encode the training data
```

```
sample = preprocess.(sample_data()) = ([11, 5, 4, 2, 5, 2, 5, 5, 5, 7, 8, 12], [11, 5, 4, 2, 5, 2, 5, 5, 5, 7, 8, 12])
encoded_sample = vocab(sample[1]) = [2, 8, 7, 5, 8, 5, 8, 8, 8, 10, 11, 3]
```


### Defining the model

With the Transformers.jl and Flux.jl, we can define the model easily. We use a Transformer with 512 hidden size and 8 head.

```julia
#define a Word embedding layer which turn word index to word vector
embed = Embed(512, length(vocab)) |> gpu
#define a position embedding layer metioned above
pe = PositionEmbedding(512) |> gpu

#wrapper for get embedding
function embedding(x)
  we = embed(x, inv(sqrt(512))) 
  e = we .+ pe(we)
	return e
end

#define 2 layer of transformer
encode_t1 = Transformer(512, 8, 64, 2048) |> gpu  
encode_t2 = Transformer(512, 8, 64, 2048) |> gpu

#define 2 layer of transformer decoder
decode_t1 = TransformerDecoder(512, 8, 64, 2048) |> gpu  
decode_t2 = TransformerDecoder(512, 8, 64, 2048) |> gpu

#define the layer to get the final output probabilities
linear = Positionwise(Dense(512, length(vocab)), logsoftmax) |> gpu

function encoder_forward(x)
  e = embedding(x)
  t1 = encode_t1(e)
  t2 = encode_t2(t1)
  return t2
end

function decoder_forward(x, m)
  e = embedding(x)
  t1 = decode_t1(e, m)
  t2 = decode_t2(t1, m)
  p = linear(t2)
	return p
end
```

Then run the model on the sample

```julia
enc = encoder_forward(encoded_sample)
probs = decoder_forward(encoded_sample, enc)
```

We can also use the`Transformers.Stack` to define the encoder and decoder so you can define multiple layer and the `xx_forwawrd` at once. See the [docs](stacks.md) for more information about the API.

### define the loss and training loop

For the last step, we need to define the loss function and training loop. We use the kl divergence for the output probability.

```julia
function smooth(et)
    sm = fill!(similar(et, Float32), 1e-6/size(embed, 2))
    p = sm .* (1 .+ -et)
    label = p .+ et .* (1 - convert(Float32, 1e-6))
    label
end

#define loss function
function loss(x, y)
  label = onehot(vocab, y) #turn the index to one-hot encoding
  label = smooth(label) #perform label smoothing
  enc = encoder_forward(x)
	probs = decoder_forward(y, enc)
  l = logkldivergence(label[:, 2:end, :], probs[:, 1:end-1, :])
  return l
end

#collect all the parameters
ps = params(embed, pe, encode_t1, encode_t2, decode_t1, decode_t2, linear)
opt = ADAM(1e-4)

#function for created batched data
using Transformers.Datasets: batched

#flux function for update parameters
using Flux: gradient
using Flux.Optimise: update!

#define training loop
function train!()
  @info "start training"
  for i = 1:2000
    data = batched([sample_data() for i = 1:32]) #create 32 random sample and batched
		x, y = preprocess.(data[1]), preprocess.(data[2])
    x, y = vocab(x), vocab(y)#encode the data
    x, y = todevice(x, y) #move to gpu
    l = loss(x, y)
    grad = gradient(()->l, ps)
    if i % 8 == 0
    	println("loss = $l")
    end
    update!(opt, ps, grad)
  end
end
```

```julia
train!()
```

### Test our model

After training, we can try to test the model.

```julia
using Flux: onecold
function translate(x)
    ix = todevice(vocab(preprocess(x)))
    seq = [startsym]

    enc = encoder_forward(ix)

    len = length(ix)
    for i = 1:2len
        trg = todevice(vocab(seq))
        dec = decoder_forward(trg, enc)
        #move back to gpu due to argmax wrong result on CuArrays
        ntok = onecold(collect(dec), labels)
        push!(seq, ntok[end])
        ntok[end] == endsym && break
    end
  seq[2:end-1]
end
```

```julia
translate([5,5,6,6,1,2,3,4,7, 10])
```

```
10-element Array{Int64,1}:
  5
  5
  6
  6
  1
  2
  3
  4
  7
 10
```

The result looks good!

