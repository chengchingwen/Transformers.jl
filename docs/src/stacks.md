# Transformers.Stacks
Helper struct and DSL for stacking functions/layers.

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


## The Stack NNTopo DSL

Since the `TransformerDecoder` require more than one input, it's not convenient to use with `Chain`. Therefore, we implement a very simple 
DSL(Domain Specific Language) to handle the function structure. You can use the `@nntopo` macro to define the structure then call the function 
with the given model.

## NNTopo Syntax

we call the DSL NNTopo for "Neural Network Topology", but actually it is just used to define where the input & output should be in a sequence of 
function, or the complex version of the `|>` function in Julia.

### "Chain" the functions

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


### Loop unrolling

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

### Multiple argument & jump connection

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

### Specify the variables you want

Notice that we use a `:` to seperate the input/output variables name for each function call, if the `:` is not present, we will by default assume 
the output variables are all the inputs of the next function call. i.e. `x => (t1, t2) => y` is equal to `x => (t1, t2):(t1, t2) => y`. 

We can also return multiple variables, so the complete syntax can be viewed as:
    
        (input arguments):(function1 inputs) => (function1 outputs):(function2 inputs):(function2 outputs) => .... => (function_n outputs):(return variables)

### Interpolation

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

### Nested Structure

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

### Collect Variables

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

## Stack

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


