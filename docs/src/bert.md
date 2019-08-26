# Transformers.BidirectionalEncoder
Implementation of BERT model

## Get Pretrain

One of the pleasant feature BERT give us is that it make using a large pre-trained model on a language task possible. To use a pretrained model with `Transformers.jl`, you can either:

1. Use the public released weight with `pretrain""` as show in the [pretrain section](pretrain.md)

2. Convert the TensorFlow checkpoint file with [`BidirectionalEncoder.tfckpt2bson`](@ref)(need `TensorFlow.jl` installed). As long as the checkpoint is produced by the origin bert released by google, it should work without any issue. 

## Finetuning

After getting the pretrain weight in Julia. We are going to finetune the model on your dataset. The bert model is also a Flux layer, so 
training bert is just like training other Flux model (i.e. all the usage should be compatibled with Flux's API)

## API reference

```@docs
Bert
BidirectionalEncoder.WordPiece
BidirectionalEncoder.bert_cased_tokenizer
BidirectionalEncoder.bert_uncased_tokenizer
BidirectionalEncoder.tfckpt2bson
BidirectionalEncoder.load_bert_pretrain
masklmloss
bert_pretrain_task
BidirectionalEncoder.recursive_readdir
```
