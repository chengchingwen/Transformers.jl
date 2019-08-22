# Transformers.Pretrain
Functions for download and loading pretrain models.

## using Pretrains

For GPT and BERT, we provide a simple api to get the released pretrain weight and load them into our Julia version Transformer implementation. 

```julia
using Transformers
using Transformers.Pretrain
using Transformers.GenerativePreTrain
using Transformers.BidirectionalEncoder

# disable cli download check
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

#load everything in the pretrain model
bert_model, wordpiece, tokenizer = pretrain"Bert-uncased_L-12_H-768_A-12" 

#load model weight only
gpt_model = pretrain"gpt-OpenAIftlm:gpt_model"

#show the loaded model
show(bert_model)
show(gpt_model)
```

```
TransformerModel{Bert}(
  embed = CompositeEmbedding(tok = Embed(768), segment = Embed(768), pe = PositionEmbedding(768, max_len=512), postprocessor = Positionwise(LayerNorm(768), Dropout{Float64}(0.1, true))),
  transformers = Bert(layers=12, head=12, head_size=64, pwffn_size=3072, size=768),
  classifier = 
    (
      pooler => Dense(768, 768, tanh)
      masklm => (
        transform => Chain(Dense(768, 768, gelu), LayerNorm(768))
        output_bias => TrackedArray{…,Array{Float32,1}}
      )
      nextsentence => Chain(Dense(768, 2), logsoftmax)
    )
)
TransformerModel{Gpt}(
  embed = CompositeEmbedding(tok = Embed(768), pe = PositionEmbedding(768, max_len=512)),
  transformers = Gpt(layers=12, head=12, head_size=64, pwffn_size=3072, size=768)
)
```

The `pretrain"<model>-<model-name>:<item>"` string with `pretrain` prefix will load the specific item from a known pretrain file (see the list below). 
The `<model>` is matched case insensitively, so not matter `bert`, `Bert`, `BERT`, or even `bErT` will find the BERT pretrain model. On the other hand, 
the `<model-name>`, and `<item>` should be exactly the one on the list. See `example`.

Currently support pretrain:

model   | model name                          | support items                       
-------:|:------------------------------------|:------------------------------------
GPT     | `OpenAIftlm`                        | gpt_model, bpe, vocab, tokenizer
Bert    | `uncased_L-24_H-1024_A-16`          | bert_model, wordpiece, tokenizer
Bert    | `wwm_cased_L-24_H-1024_A-16`        | bert_model, wordpiece, tokenizer
Bert    | `wwm_uncased_L-24_H-1024_A-16`      | bert_model, wordpiece, tokenizer
Bert    | `multilingual_L-12_H-768_A-12`      | bert_model, wordpiece, tokenizer
Bert    | `multi_cased_L-12_H-768_A-12`       | bert_model, wordpiece, tokenizer
Bert    | `chinese_L-12_H-768_A-12`           | bert_model, wordpiece, tokenizer
Bert    | `cased_L-24_H-1024_A-16`            | bert_model, wordpiece, tokenizer
Bert    | `cased_L-12_H-768_A-12`             | bert_model, wordpiece, tokenizer
Bert    | `uncased_L-12_H-768_A-12`           | bert_model, wordpiece, tokenizer


If you don't find a public pretrain you want on the list, please fire an issue.

See `example` folder for the complete example.


## API reference

```@autodocs
Modules=[Transformers.Pretrain]
Order = [:type, :function, :macro]
```
