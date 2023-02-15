# Transformers.HuggingFace

Module for loading pre-trained model from HuggingFace.

!!! info
    We provide a set of API to download & load a pretrain model from huggingface hub. This is mostly manually done, so we
     only have a small set of available models. The most practical way to check if a model is available in Transformers
     is to run the [`HuggingFaceValidation` code in the example folder](https://github.com/chengchingwen/Transformers.jl/tree/master/example/HuggingFaceValidation),
     which use `PyCall.jl` to load the model in both Python and Julia. Open issues/PRs if you find a model you want is
	 not supported here.


There are basically 3 main api for loading the model, [`HuggingFace.load_config`](@ref),
[`HuggingFace.load_tokenizer`](@ref), [`HuggingFace.load_model`](@ref). These are the underlying function of
the [`HuggingFace.@hgf_str`](@ref) macro. You can get a better control of the loading process.


We can load a specific config of a specific model, no matter it's actually supported by Transformers.jl.

```julia-repl
julia> load_config("google/pegasus-xsum")
Transformers.HuggingFace.HGFConfig{:pegasus, JSON3.Object{Vector{UInt8}, Vector{UInt64}}, Dict{Symbol, Any}} with 45 entries:
  :use_cache                       => true
  :d_model                         => 1024
  :scale_embedding                 => true
  :add_bias_logits                 => false
  :static_position_embeddings      => true
  :encoder_attention_heads         => 16
  :num_hidden_layers               => 16
  :encoder_layerdrop               => 0
  :num_beams                       => 8
  :max_position_embeddings         => 512
  :model_type                      => "pegasus"
  ⋮                                => ⋮

```

This would give you all value available in the downloaded configuration file. This might be enough for a some model,
 but there are other model that use the default value hard coded in their python code.


Sometime you would want to add/overwrite
 some of the value. This can be done be calling `HGFConfig(old_config; key_to_update = new_value, ...)`. These is used
 primary for customizing model loading. For example, you can load a `bert-base-cased` model for sequence classification
 task. However, if you directly load the model:

```julia-repl
julia> bert_model = hgf"bert-base-cased:ForSequenceClassification";

julia> bert_model.cls.layer
Dense(W = (768, 2), b = true)
```

The model is default creating model for 2 class of label. So you would need to load the config and update the field
 about number of labels and create the model with the new config:

```julia-repl
julia> bertcfg = load_config("bert-base-cased");

julia> bertcfg.num_labels
2

julia> mycfg = HuggingFace.HGFConfig(bertcfg; num_labels = 3);

julia> mycfg.num_labels
3

julia> _bert_model = load_model("bert-base-cased", :ForSequenceClassification; config = mycfg);

julia> _bert_model.cls.layer
Dense(W = (768, 3), b = true)

```

All config field name follow the same name as huggingface, so you might need to read their document for what
 is available. However, not every configuration work in Transformers.jl. It's better to check [the source
 `src/huggingface/implementation`](https://github.com/chengchingwen/Transformers.jl/tree/master/src/huggingface/implementation). All supported models would need to overload the `load_model` and provided an implementation in Julia to be
 workable.


For the tokenizer, `load_tokenizer` is basically the same as calling with `@hgf_str`. Currently providing customized
 config doesn't change much stuff. The tokenizer might also work for unsupported model because some serialize the whole
 tokenizer object, but not every model does that or they use something not covered by our implementation.

## API Reference

```@autodocs
Modules = [Transformers.HuggingFace]
Order = [:macro, :type, :function]
```
