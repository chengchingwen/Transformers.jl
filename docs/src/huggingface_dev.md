# Adding New HuggingFace Model

This is a record of what I do to port the bloom model. This log serves as an tutorial for adding new hugginface model to Transformers.jl.

## 0. Find the correct model type

Our target is to port the `bigscience/bloom` model. We first find the model on the huggingface hub (`https://huggingface.co/bigscience/bloom/tree/main`). At the "Files and versions", we can find the `config.json` and in the `model_type` field we can see the model type is `"bloom"`. We can find another model of same model type but with smaller model size, such as `bigscience/bloom-560m`, for testing.

Once we get the model type, we can locate the corresponding python code in huggingface transformers github under the `"src/transformers/models"` folder (`https://github.com/huggingface/transformers/tree/main/src/transformers/models`). The code would usually be put in a folder using model type as the name, such as [`"src/transformers/models/bloom"`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bloom). On the other hand, our code will be put inside `"src/huggingface/implementation"` and also use model type as the folder name (`https://github.com/chengchingwen/Transformers.jl/tree/master/src/huggingface/implementation/bloom`).

## 1. Porting config

The first thing we need to port is the config object. Although we are actually able to load the config without defining the config object in Julia, there are some default values hardcoded in the python code and cannot be found in the `config.json`. Thus we copy the hardcoded default values and create our own config object in julia.

Their config object is defined in `configuration_<model_type>.py` (`https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/configuration_bloom.py`) as `class <model_type>Config(PretrainedConfig)` (`BloomConfig`). The default values is either defined as the `__init__` default arguments or assigned in the function body. We copy the default values to our julia definition. Generally, there are 3 part we need to check in the python class:
1. variable [`model_type`](https://github.com/huggingface/transformers/blob/29e7a1e1834f331a4916853ecd58549ed78235d6/src/transformers/models/bloom/configuration_bloom.py#L106)
2. variable [`attribute_map`](https://github.com/huggingface/transformers/blob/29e7a1e1834f331a4916853ecd58549ed78235d6/src/transformers/models/bloom/configuration_bloom.py#L108-L111): A dictionary that map alias property name to the real property name.
3. method [`__init__`](https://github.com/huggingface/transformers/blob/29e7a1e1834f331a4916853ecd58549ed78235d6/src/transformers/models/bloom/configuration_bloom.py#L113-L149)

In `"src/huggingface/implementation/bloom/"`, we open a file `"config.jl"` and define the config object:

```julia
@hgfcfg :bloom struct HGFBloomConfig
    vocab_size::Int = 250880
    hidden_size::Int = 64
    [n_layer, num_hidden_layers]::Int = 2
    [n_head, num_attention_heads]::Int = 8
    layer_norm_epsilon::Float64 = 1e-5
    initializer_range::Float64 = 0.02
    use_cache::Bool = true
    bos_token_id::Int = 1
    eos_token_id::Int = 2
    apply_residual_connection_post_layernorm::Bool = false
    hidden_dropout::Float64 = 0.0
    attention_dropout::Float64 = 0.0
    pretraining_tp::Int = 1
    slow_but_exact::Bool = false
    clean_up_tokenization_spaces::Bool = false
end
```

The `@hgfcfg` take two argument: 1. a symbol for the model type (`:bloom`). 2. a `Base.@kwdef`-like struct definition, despite the name aliases (e.g. `[n_layer, num_hidden_layers]` where `n_layer` is the real field name and `num_hidden_layers` is an alias), with the default values copy from the python definition. The struct follow the name of `HGF<model_type>Config`. Notice that the config are not necessary used or implemented in our julia implementation.

## 2. Check the tokenizer

Currently our tokenizer does not support much modification from the loading API. Our loading API should be able to directly load most of the tokenizer. We check the tokenizer by running [our huggingface validation code](https://github.com/chengchingwen/Transformers.jl/tree/master/example/HuggingFaceValidation). The validation code use PyCall to load the huggingface transformers and directly compare the result. The validation code for tokenizer take two argument, the model name and the a corpus. You can take any corpus for the test, but we recommand using the [xnli dataset](https://github.com/facebookresearch/XNLI). We use the devset of xnli and extract all sentences in the dataset as the testing corpus, which is also available with the following `Artifacts.toml`:

```toml
[xnli_dev]
git-tree-sha1 = "fb42e6f4a4522b30eb09d21786b90617c4792113"
lazy = true

    [[xnli_dev.download]]
    sha256 = "9fa4c2b8ff5194eb3eb9cd63a264a2f1e2b9e24f5d71781d524cfe0f4b268c25"
    url = "https://gist.github.com/chengchingwen/27796c1d39efdae744e5abec94ecfdb6/raw/fb42e6f4a4522b30eb09d21786b90617c4792113.tar.gz"
```

If the validation code passed, we are fine. If the tokenizer cannot be loaded, or the code does not pass the validation, then issues should be opened.

## 3. Porting model

To port the model, we need two things: the model weights and an implementation of the model in Julia. Currently we only support loading model weights stored in pytorch pickle format with `Transformers.HuggingFace.load_state_dict` and it should be able to load the model weights (that support pytorch) without problems. On the other hand, the implementation of the model need to be done manually.

### Types

The first thing we do is checking the python implementation source code, in `"src/huggingface/implementation/bloom/modeling_<model_type>.py"` (`https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py`). We search for `class <model_type>` (`class Bloom`) in the file and we can see there are 10 matches and organized into 3 categories:

#### 1. Implementation Detail Types (`BloomGelu`, `BloomAttention`, `BloomMLP`, `BloomBlock`)

These types are part of the implementation detail of the model. We won't directly translate these types into Julia. Instead, we use/implement the julia functions/types that perform the similar operations. For example, we would directly use `NNlib.gelu` for `BloomGelu`. Usually the attention (`BloomAttention`) is the most important part we need to look at because it determine whether the model requires a different implementation. We can see that `BloomAttention` use an attention variant with alibi position embedding. The attention variants would usually be implemented with [NeuralAttentionlib.jl (NAlib)](https://github.com/chengchingwen/NeuralAttentionlib.jl). If the components for implementing the attention variant are not found in NAlib, open issue/PR in NAlib. Then we would use the components provided from NAlib to implement the attention operation/type.

#### 2. Model Type (`BloomPreTrainedModel`, `BloomModel`)

The `<model_type>PreTrainedModel` (`BloomPreTrainedModel`) is a python class for inheritance, so we will convert it into a abstract type in julia like (for illustration only):

```julia
abstract type HGFBloomPreTrainedModel <: HGFPreTrainedModel end
```

The `"HGF"` prefix indicate that this type correspond to a python class in huggingface transformers. The `<model_type>Model` (`BloomModel`) is the base model for each task-specific model. We'll translate it into a julia type `HGF<model_type>Model` (`HGFBloomModel`). For example, the julia type for bloom model is:

```julia
struct HGFBloomModel{E, D} <: HGFBloomPreTrainedModel
    embed::E
    decoder::D
end
@functor HGFBloomModel

(model::HGFBloomModel)(nt::NamedTuple) = model.decoder(model.embed(nt))
```

This specify the fields of bloom, mark it as a layer, and define the interface. The actual computation are carry out by the embedding and decoder (`embed::E` and `decoder::D`), so at this point we can't even tell what computation does bloom perform. This means we can actually directly use the constructor to create a model that is not a "bloom" but has the `HGFBloomModel` type, and this should usually be avoided. The "real" bloom model is constructed by the `Transformers.HuggingFace.load_model` api, which would be discussed in the following sections.

#### 3. Task Specific Types (`BloomForCausalLM`, `BloomForSequenceClassification`, `BloomForTokenClassification`, `BloomForQuestionAnswering`)

Currently, each task specific class would also be translated into a julia type having the based model (`model` field) and a classifier (`cls` field). For example, the `HGFBloomForCausalLM` can be defined as:

```julia
struct HGFBloomForCausalLM{M, C} <: HGFBloomPreTrainedModel
	model::M
	cls::C
end
@functor HGFBloomForCausalLM

(model::HGFBloomForCausalLM)(nt::NamedTuple) = model.cls(model.model(nt))
```

Similar to `HGFBloomModel`, this definition does not include the computation and require to use `Transformers.HuggingFace.load_model` for the correct construction.

### Implementation

There are basically 3 steps to implement a new model:

#### 1. Define Types

We provide a macro `Transformers.HuggingFace.@hgfdef` that generate the model types for us. For example, the above Julia types is defined with:

```julia
@hgfdef Bloom (
    Model => (embed, decoder),
    ForCausalLM,
)
```

which would be expanded to:

```julia
const HGFBloomPreTrainedModel = HGFPreTrained{:bloom}

struct HGFBloomModel{EMBED, DECODER} <: HGFPreTrained{:bloom, :model}
    embed::EMBED
    decoder::DECODER
end
@functor HGFBloomModel

@inline function Transformers.HuggingFace.hgf_model_forward(model::HGFBloomModel, nt::NamedTuple)
	return model.decoder(model.embed(nt))
end

struct HGFBloomForCausalLM{MODEL, CLS} <: HGFPreTrained{:bloom, :forcausallm}
    model::MODEL
    cls::CLS
end
@functor HGFBloomForCausalLM

@inline function Transformers.HuggingFace.hgf_model_forward(model::HGFBloomForCausalLM, nt::NamedTuple)
	return model.cls(model.model(nt))
end
```

The `HGFPreTrained` is the real abstract type for our huggingface models. It take two type parameters -- `HGFPreTrained{model_type, task}` (e.g. `HGFPreTrained{:bloom, :forcausallm}`). This allow use to query the supported model through `subtypes`. Moreover, we can define behaviors for all model of a specific `task`. For example, `Transformers.HuggingFace.hgf_model_loss(model::HGFPreTrained{M, :forcausallm} where M)` return the loss function for all model for causal LM task.

The `@hgfdef` macro take 3 arguments: the model type (`:bloom`), capitalized name (`Bloom`), and a tuple of tasks. If the model type is omitted, it will use the lowercase of capitalized name. Each task in the tuple is a pair of task name and the forward function body. It collect all `getproperty` on `model` in the function body for the field names of the type. If the function body is omitted, it will use `(model, cls)` as default. If the function body is a tuple of field names, it will convert them into a chain of function call (e.g. `(embed, decoder)` to `model.decoder(model.embed(nt))`).

#### 2. Overload `Transformers.HuggingFace.load_model`

As mentioned above, the `Transformers.HuggingFace.load_model` serves as the actual constructor for our huggingface models. `load_model` is dispatch on the type itself. For example:

```julia
function load_model(_type::Type{HGFBloomModel}, cfg, state_dict, prefix)
    embed = load_model(_type, CompositeEmbedding, cfg, state_dict, prefix)
    decoder = load_model(_type, TransformerBlock, cfg, state_dict, prefix)
    return HGFBloomModel(embed, decoder)
end

function load_model(_type::Type{<:HGFBloomPreTrainedModel}, ::Type{<:CompositeEmbedding}, cfg, state_dict, prefix)
    vocab_size, dims = cfg[:vocab_size], cfg[:hidden_size]
    factor = Float32(cfg[:initializer_range])
    token_embed = _load_embed(state_dict, joinname(prefix, "word_embeddings"), vocab_size, dims, factor)
    embed = CompositeEmbedding(token = token_embed)
    ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(prefix, "word_embeddings_layernorm"))
    return Layers.Chain(embed, ln)
end

...
```

`load_model(_type::Type{HGFBloomModel}, cfg, state_dict, prefix)` isa the main function to overload for loading `HGFBloomModel`. `cfg` is the `HGFConfig` of the loaded model. `state_dict` is the model weights loaded by `Transformers.HuggingFace.load_state_dict` which is a dictionay map from flattened python property names to the weight array (e.g. `"word_embeddings.weight" => Float16[-0.00987 -0.00481 … `). Since our model does not follow the same hierarchy and field names, we use `prefix` and `Transformers.HuggingFace.joinname(prefix, another_field_name)` to mimic the traversal. We use the corresponding information to reconstruct the correct model with `Transformers.Layers`. The weight access is hide in the `_load_embed` function, which is defined as:

```julia
function _load_embed(state_dict, prefix, w_init, pad_idx0 = nothing)
    embedding = getweight(Layers.Embed, state_dict, joinname(prefix, "weight")) do
        weight = w_init()
        if !isnothing(pad_idx0)
            weight[:, pad_idx0 + 1] .= 0
        end
        return weight
    end
    return Layers.Embed(embedding)
end
```

All weight accesses are done with `getweight` which check if the name (`joinname(prefix, "weight")`) is present in the `state_dict`, and if not, create a new array and store it in the `state_dict` with the name. This allow us to handle the case that some weight are missing, like using the pretrained model for finetuning on a new task. Besides, we need to overload `basemodelkey(::Type{<:HGFPreTrained{:bloom}}) = :transformer` for loading the model correctly. This is equivalent to the [`base_model_prefix`](https://github.com/huggingface/transformers/blob/fa21ead73db473d88f8eca1ec244aba776fd9047/src/transformers/models/bloom/modeling_bloom.py#L446) class variable of `BloomPreTrainedModel` in `"modeling_bloom.py"` (`https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py`).

#### 3. Overload `Transformers.HuggingFace.get_state_dict`

After finishing the loader, we also need to overload the `Transformers.HuggingFace.get_state_dict` which extract all weights in a model and store in a flat dictionary. Essentially, `model::HGFBloomModel == load_model(HGFBloomModel, cfg, get_state_dict(model))`. `get_state_dict` is dispatch on the model. For example:

```julia
function get_state_dict(m::HGFBloomModel, state_dict, prefix)
    get_state_dict(HGFBloomModel, m.embed[1], state_dict, prefix)
    get_state_dict(HGFBloomModel, m.embed[2], state_dict, joinname(prefix, "word_embeddings_layernorm"))
    get_state_dict(HGFBloomModel, m.decoder[1], state_dict, prefix)
    get_state_dict(HGFBloomModel, m.decoder[2], state_dict, joinname(prefix, "ln_f"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFBloomPreTrainedModel}, m::CompositeEmbedding, state_dict, prefix)
    get_state_dict(p, m.token, state_dict, joinname(prefix, "word_embeddings"))
    return state_dict
end

get_state_dict(_, m::Layers.Embed, state_dict, prefix) = get_state_dict(m, state_dict, prefix)
function get_state_dict(m::Layers.Embed, state_dict, prefix)
    state_dict[joinname(prefix, "weight")] = m.embeddings'
    return state_dict
end
...
```

### Validation

After implementing the model, we use the same [script](https://github.com/chengchingwen/Transformers.jl/tree/master/example/HuggingFaceValidation) mentioned in the tokenizer part to check if our model perform the same computation as Python.
