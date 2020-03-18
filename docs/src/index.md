# Transformers.jl

*Julia implementation of Transformers models*

This is the documentation of `Transformers`: The Julia solution for using Transformer models based on [Flux.jl](https://fluxml.ai/)


## Installation

In the Julia REPL:

```jl
julia> ]add Transformers
```

For using GPU, make sure `CuArrays` is runable on your computer:

```jl
julia> ]add CuArrays; build
```


## Implemented model
You can find the code in `example` folder.

-   [Attention is all you need](https://arxiv.org/abs/1706.03762)
-   [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
-   [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)


## Example

Using pretrained Bert with `Transformers.jl`.

```julia
using Transformers
using Transformers.Basic
using Transformers.Pretrain

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
vocab = Vocabulary(wordpiece)

text1 = "Peter Piper picked a peck of pickled peppers" |> tokenizer |> wordpiece
text2 = "Fuzzy Wuzzy was a bear" |> tokenizer |> wordpiece

text = ["[CLS]"; text1; "[SEP]"; text2; "[SEP]"]
@assert text == [
    "[CLS]", "peter", "piper", "picked", "a", "peck", "of", "pick", "##led", "peppers", "[SEP]", 
    "fuzzy", "wu", "##zzy",  "was", "a", "bear", "[SEP]"
]

token_indices = vocab(text)
segment_indices = [fill(1, length(text1)+2); fill(2, length(text2)+1)]

sample = (tok = token_indices, segment = segment_indices)

bert_embedding = sample |> bert_model.embed
feature_tensors = bert_embedding |> bert_model.transformers
```

# Module Hierarchy

- [Transformers.Basic](./basic.md)
Basic functionality of Transformers.jl, provide the Transformer encoder/decoder implementation and other convenient function.

- [Transformers.Pretrain](./pretrain.md)
Functions for download and loading pretrain models.

- [Transformers.Stacks](./stacks.md)
Helper struct and DSL for stacking functions/layers.

- [Transformers.Datasets](./datasets.md)
Functions for loading some common Datasets

- [Transformers.GenerativePreTrain](./gpt.md)
Implementation of gpt-1 model

- [Transformers.BidirectionalEncoder](./bert.md)
Implementation of BERT model


## Outline

```@contents
Pages = [
  "tutorial.md",
  "basic.md",
  "stacks.md",
  "pretrain.md",
  "gpt.md",
  "bert.md",
  "datasets.md",
]
Depth = 3
```
