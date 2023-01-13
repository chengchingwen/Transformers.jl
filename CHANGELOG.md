% ChangeLog

# Changes from 0.1.x to 0.2.0

v0.2 is a rewrite of the whole package. Most layers and API in 0.1 is removed or changed. Some of them are replaced
 with new one. The basic policy is, if a functionality is achievable with a well-maintained package easily, or there
 isn't much gain by self-hosting/maintaining it, then we remove the functionality from Transformers.jl.


Here is list of the changes with brief explanation:

## Module Changes

### Transformers.Pretrain

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


### Transformers.Stacks

The `Stacks` module is entirely removed. `Stacks` provide a small DSL for creating nontrivial `Chain` of layers.
 However, the DSL isn't intuitive enough and it also doesn't seems worth maintaining a DSL. We don't provide
 direct replacement for this, but for the specific use case of building transformer models, we have a few new
 constructors/layers in `Transformers.Layers`.


### Transformers.Basic

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


### Transformers.Layers (new)

The `Layers` module is a new module introduced in v0.2.0. It provide a set layer types for construct transformer
 model variants.


### Transformers.TextEncoders (new)

The `TextEncoders` module is a new module introduced in v0.2.0. Basically all old functionality about text preprocessing
 are moved to this module, including `WordPiece`, `Unigram`, `BertTextEncoder`, `GPT2TextEncoder`, etc.

### Transformers.BidirectionalEncoder / Transformers.GenerativePreTrain

These modules are removed since we are switching to the `Transformers.HuggingFace`. The text encoder are moved to
 `Transformers.TextEncoders`. Weight loading and conversion functionality are removed. If you need that, use
 the tools that huggingface transformers python package provided and make sure the model can be loaded with pytorch.
 Then we can use the weight in pytorch format.


### Transformers.HuggingFace
