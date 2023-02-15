# ChangeLogs (from 0.1.x to 0.2.0)

v0.2 is a rewrite of the whole package. Most layers and API in 0.1 is removed or changed. Some of them are replaced
 with new one. The basic policy is, if a functionality is achievable with a well-maintained package easily, or there
 isn't much gain by self-hosting/maintaining it, then we remove the functionality from Transformers.jl.


Here is list of the changes with brief explanation:

## Transformers.Pretrain

The `Pretrain` module is entirely removed, due to the duplication of functionality v.s. `Transformers.HuggingFace`.
 We do not host the small list of the origin official released pretrained weights anymore. All use that require a
 pretrained weight should refer to `HuggingFace` module. This is a table of the old pretrain name and corresponding
 huggingface model name:

| old pretrain name              | corresponding huggingface model name    |
|--------------------------------|-----------------------------------------|
| `cased_L-12_H-768_A-12`        | `bert-base-cased`                       |
| `uncased_L-12_H-768_A-12`      | `bert-base-uncased`                     |
| `chinese_L-12_H-768_A-12`      | `bert-base-chinese`                     |
| `multi_cased_L-12_H-768_A-12`  | `bert-base-multilingual-cased`          |
| `multilingual_L-12_H-768_A-12` | `bert-base-multilingual-uncased`        |
| `cased_L-24_H-1024_A-16`       | `bert-large-cased`                      |
| `uncased_L-24_H-1024_A-16`     | `bert-large-uncased`                    |
| `wwm_cased_L-24_H-1024_A-16`   | `bert-large-cased-whole-word-masking`   |
| `wwm_uncased_L-24_H-1024_A-16` | `bert-large-uncased-whole-word-masking` |
| `scibert_scivocab_cased`       | `allenai/scibert_scivocab_cased`        |
| `scibert_scivocab_uncased`     | `allenai/scibert_scivocab_uncased`      |
| `scibert_basevocab_cased`      | N/A                                     |
| `scibert_basevocab_uncased`    | N/A                                     |
| `OpenAIftlm`                   | `openai-gpt`                            |


## Transformers.Stacks

The `Stacks` module is entirely removed. `Stacks` provide a small DSL for creating nontrivial `Chain` of layers.
 However, the DSL isn't intuitive enough and it also doesn't seems worth maintaining a DSL. We don't provide
 direct replacement for this, but for the specific use case of building transformer models, we have a few new
 constructors/layers in `Transformers.Layers`.


## Transformers.Basic

The `Basic` module is now destructed and most of the elements in `Basic` is separated to other module/package.

1. `Transformer` and `TransformerDecoder`: The `Transformer`/`TransformerDecoder` layer is replaced with the new
    implementation in `Layers` (the `Layers.TransformerBlock`, `Layers.TransformerDecoderBlock`, and friends).
2. `MultiheadAttention`: The implementation of attention operations are move out to
    [NeuralAttentionlib](https://github.com/chengchingwen/NeuralAttentionlib.jl). In NeuralAttentionlib, we can use
    `multihead_qkv_attention` to do the same computation. Since most transformer variant only use a modified version
    of self or cross attention, we do not provied the `MultiheadAttention` layer type. One should be able to redefine
    the `MultiheadAttention` layer type with Flux and NeuralAttentionlib easily. For example:

   ```julia
   using Flux, Functors
   using NeuralAttentionlib: multihead_qkv_attention, CausalMask

   struct MultiheadAttention{Q,K,V,O}
       head::Int
       future::Bool
       iqproj::Q
       ikproj::K
       ivproj::V
       oproj::O
   end
   @functor MultiheadAttention (iqproj, ikproj, ivporj, oproj)
   MultiheadAttention(head, hidden_size, head_size; future = true) =
       MultiheadAttention(head, future,
           Dense(hidden_size, head_size * head),
           Dense(hidden_size, head_size * head),
           Dense(hidden_size, head_size * head),
           Dense(head_size * head, hidden_size),
       )

   (mha::MultiheadAttention)(q, k, v) = mha.oproj(multihead_qkv_attention(mha.head,
       mha.iqproj(q), mha.ikproj(k), mha.ivproj(v), mha.future ? nothing : CausalMask()))
   ```

3. `TransformerModel`: This is just a Flux layer with embedding layer, transformer layer, and classifier layer
     bundle together. One can define this easily with Flux/Functors API, thus removed.
4. `Positionwise`, `PwFFN`, and `@toNd`: This was originally designed for applying `Flux.Dense` on 3-dim arrays,
    but since `Flux.Dense` support multi-dim input now. This isn't needed and thus removed.
5. `EmbeddingDecoder`: Replaced with `Layers.EmbedDecoder`. Name change and support extra trainable `bias` parameter.
6. `PositionEmbedding`: This is replace with `Layers.SinCosPositionEmbed` and `Layers.FixedLenPositionEmbed` for
    the old `trainable` keyword argument setting.
7. `crossentropy` with masking: We extend `Flux.logitcrossentropy` and `Flux.crossentropy` with 3-args
    input (the prediction, label, and mask) and 4-args input (`sum` or `mean`, prediciton, label, and mask).
8. `kldivergence`: In our use case (i.e. training language model), this is equivalent to cross-entropy, thus removed.
9. `logcrossentropy`/`logkldivergence`: This is a fault design. Originally I would put a `logsoftmax` at the head of
    the prediction head. However, that is not only unnecessary but also increasing the amount of memory needed.
    One should use `Flux.logitcrossentropy` without the `logsoftmax` directly.
10. `Vocabulary`: Replaced with `TextEncodeBase.Vocab`.
11. `with_firsthead_tail`/`segment_and_concat`/`concat`: These can be implemented with `TextEncodeBase.SequenceTemplate`
     and friends thus removed.
12. `getmask`: The attention mask functionality is moved to NeuralAttentionlib. Manually construct attention mask
     should use constructor in `NeuralAttentionlib.Masks`.


## Transformers.Layers (new)

The `Layers` module is a new module introduced in v0.2.0. It provide a set layer types for construct transformer
 model variants.


## Transformers.TextEncoders (new)

The `TextEncoders` module is a new module introduced in v0.2.0. Basically all old functionality about text preprocessing
 are moved to this module, including `WordPiece`, `Unigram`, `BertTextEncoder`, `GPT2TextEncoder`, etc.

## Transformers.BidirectionalEncoder / Transformers.GenerativePreTrain

These modules are removed since we are switching to the `Transformers.HuggingFace` for the pretrained model. The text
 encoder are moved to `Transformers.TextEncoders`. Weight loading and conversion functionality are removed. If you
 need that, use the tools that huggingface transformers python package provided and make sure the model can be loaded
 with pytorch. Then we can use the weight in pytorch format.


## Transformers.HuggingFace

The changes in `Transformers.HuggingFace` are mainly about the configurations and models. The tokenizer/textencoder part
 are mostly the same, except the process functions.

### Configuration

For the configuration, the loading mechanism is changed. In previous version, each model type need to define a specific
 `HGF<XXModelType>Config` struct where `XXModelType` is the model type name. The reason for that is, for some reason,
 huggingface transformers doesn't serialize all the configuration values into the file, but rely on their constructor
 with pre-defined default values instead. As a result, some model only need the configuration file, while some need the
 python code for the defaults as well. The hgf config struct was more like a interal data carrier. You usually
 won't (and actually can't) manipulate the model with it.


In v0.2, we tried to make the process for adding model more automatic, and enable the ability to build model with
 different configurations. The struct for holding the configuration is now changed to a parametric struct depending
 on a `Symbol` parameter specifying the model type (e.g. `HGFConfig{:bert}`). With this, the specific
 `HGF<XXModelType>config` can be constructed on the fly. The `HGFConfig` has 2 field, one for storing the read-only
 deserialized object loaded from the configuration file, and another for the overwritten values. This should turn the
 config struct into a user level interface.


### Model

For the model part, the main change is that we do not make a 1-1 mapping between the python model/layer class and our
 julia layer struct. When one wants to add a new model type, there are actually 2 things need to be done. One is
 defining a model forward method that can do the same computation as the python model, and another is defining a
 mapping between the python model and the julia model (so that the model parameters/weights can be transferred between
 2 language). In the previous version, we chose to make a 1-1 mapping between the model, so that the parameters/weights
 loading process can be fully automatic. However, for some reason, huggingface transformers is not reusing their
 attention or transformer implementation for each model type. Which means for different model type, even if they are
 actually doing the same computation (i.e. the computation graph is the same), the model layout can be different
 (e.g. consider the differences between `Chain(Chain(dense1, dense2), dense3)` and `Chain(dense1, dense2, dense3)`).
 As a result, these make implementing the model forward method a real pain, and also it's hard to apply optimizations.


We noticed that the model forward method is more important and difficult than the model mapping. On the other hand,
 though manually defining model mapping is tedious, it's less prone to go wrong. So instead of making a 1-1 mapping for
 fully automatic model loading, we choose to reduce the work needed for forward method. In v0.2, the attention
 implementation is switched to NeuralAttentionlib's modulated implementation and we build all internal layers with layer
 from `Transformers.Layers`. As a result, layers like `FakeTH<XXLayer>` or `HGF<XXModelType>Attention/MLP/...` are
 removed, only the outer-most types remain (e.g. `HGFBertModel`, `HGFGPT2LMHeadModel`...).


Since we want to make it possible to finetune a pretrained model on new dataset/task easily, the model loading would
 be a combination of initialization and parameters/weights loading. In normal Flux workflow, you would build a complete
 new model and then inplace load the parameter/weight values into the specific layers/arrays in the model. In v0.2, we
 combine the 2 step into one `load_model` function, which take the model type, configuration, and a state dictionary (
 the term comes from PyTorch, which is a `OrderedDict` of variable names to weights). `load_model` would either
 lookup variable from the state dictionary, or initialize with configuration, recursively. As a result,
 `load_model!` is removed.


## Behavior Changes

* All text encoder (including `HuggingFace` one) process function returned `NamedTuple`: Some field name changed,
   `tok` => `token`, `mask` => `attention_mask`.
* Most layer/model from Transformers.jl would be taking and returning `NamedTuple`.
* For `HuggingFace` model: All input is basically `NamedTuple`. The returned `NamedTuple` field name from the forward
   method is also changed.
