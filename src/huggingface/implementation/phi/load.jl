using ..Layers
using ..Layers: CompositeEmbedding, SelfAttention
using ChainRulesCore
using Functors
using Static

using NeuralAttentionlib
using NeuralAttentionlib: WithScore


abstract type HGFPhiPreTrainedModel <: HGFPreTrainedModel end

struct HGFPhiModel{E, DEC} <: HGFPhiPreTrainedModel
    embed::E
    decoder::DEC
end
@functor HGFPhiModel

(model::HGFPhiModel)(nt::NamedTuple) = model.decoder(model.embed(nt))

for T in :[
    HGFPhiForCausalLM,
    # HGFPhiForSequenceClassification,
].args
    @eval begin
        @hgfdefmodel $T HGFPhiPreTrainedModel
    end
end

basemodelkey(::Type{<:HGFPhiPreTrainedModel}) = :model
isbasemodel(::Type{<:HGFPhiModel}) = true
isbasemodel(::Type{<:HGFPhiPreTrainedModel}) = false

get_model_type(::Val{:phi}) = (
    model = HGFPhiModel,
    forcausallm = HGFPhiForCausalLM,
)


function load_model(_type::Type{HGFPhiPreTrainedModel}, cfg, state_dict, prefix)
    embed = load_model(HGFLlamaModel, CompositeEmbedding, cfg, state_dict, prefix)
    decoder = load_model(_type, TransformerBlock, cfg, state_dict, prefix)
    return HGFPhiModel(embed, decoder)
end

function load_model(_type::Type{HGFPhiForCausalLM}, cfg, state_dict, prefix)
    model = load_model(HGFPhiPreTrainedModel, cfg, state_dict, joinname(prefix, "model"))
    if cfg[:tie_word_embeddings]
        embedding = model.embed.token.embeddings
    else
        vocab_size, dims, factor = cfg[:vocab_size], cfg[:hidden_size], Float32(cfg[:initializer_range])
        embedding = getweight(weight_init(vocab_size, dims, factor), Layers.Embed,
                              state_dict, joinname(prefix, "lm_head.weight"))
    end
    lmhead = Layers.EmbedDecoder(Layers.Embed(embedding))
    return HGFLlamaForCausalLM(model, Layers.Branch{(:logit,), (:hidden_state,)}(lmhead))
end

function load_model(_type::Type{<:HGFPhiForCausalLM}, ::Type{<:SelfAttention}, cfg, state_dict, prefix)
    head, dims = cfg[:num_attention_heads], cfg[:hidden_size]
    @assert dims % head == 0 "The hidden size is not a multiple of the number of attention heads."
    head_dims = div(dims, head)
    kv_head = something(cfg[:num_key_value_heads], head)
    grouped_attn = kv_head != head
    @assert head % kv_head == 0 "The number of query is not dividable by the number of key/value groups"
    return_score = cfg[:output_attentions]
    factor = Float32(cfg[:initializer_range])
    @assert isnothing(cfg[:rope_scaling]) "Scaling Rotary Embedding is not support yet"
    @show (head_dims, head, kv_head)
    q_weight = getweight(weight_init(dims, dims, factor), Array,
                         state_dict, joinname(prefix, "q_proj.weight"))
    q_bias = getweight(weight_init(dims, dims, factor), Array,
                         state_dict, joinname(prefix, "q_proj.bias"))
    k_weight = getweight(weight_init(dims, kv_head * head_dims, factor), Array,
                         state_dict, joinname(prefix, "k_proj.weight"))
    k_bias = getweight(weight_init(dims, kv_head * head_dims, factor), Array,
                         state_dict, joinname(prefix, "k_proj.bias"))
    v_weight = getweight(weight_init(dims, kv_head * head_dims, factor), Array,
                         state_dict, joinname(prefix, "v_proj.weight"))
    v_bias = getweight(weight_init(dims, kv_head * head_dims, factor), Array,
                         state_dict, joinname(prefix, "v_proj.bias"))
    o_weight = getweight(weight_init(dims, dims, factor), Array, state_dict, joinname(prefix, "dense.weight"))
    o_bias = getweight(weight_init(dims, dims), Array, state_dict, joinname(prefix, "dense.bias"))
    qkv_proj = Layers.Fork(Layers.Dense(q_weight, q_bias), Layers.Dense(k_weight, k_bias), Layers.Dense(v_weight, v_bias))
    o_proj = Layers.Dense(o_weight, o_bias)
    if grouped_attn
        op = CausalLlamaRoPEGroupedQueryAttenOp(head, kv_head)
    else
        op = CausalGPTNeoXRoPEMultiheadQKVAttenOp(head_dims, head)
    end
    return_score && (op = WithScore(op))
    return SelfAttention(op, qkv_proj, o_proj)
end


function load_model(::Type{<:HGFPhiPreTrainedModel}, ::Type{<:Layers.LayerNorm}, cfg, state_dict, prefix)
    dims = cfg[:hidden_size]
    ln_ϵ = Float32(cfg[:layer_norm_eps])
    ln_weight = getweight(one_init(dims), Array, state_dict, joinname(prefix, "weight"))
    ln_bias = getweight(zero_init(dims), Array, state_dict, joinname(prefix, "bias"))
    return Layers.LayerNorm(ln_weight, ln_bias, ln_ϵ)
end

function load_model(_type::Type{<:HGFPhiPreTrainedModel}, ::Type{<:TransformerBlock}, cfg, state_dict, prefix)
    n = cfg[:num_hidden_layers]
    collect_output = cfg[:output_attentions] || cfg[:output_hidden_states]
    blocks = []
    for i = 1:n
        lprefix = joinname(prefix, :layers, i-1)

        ln = load_model(HGFPhiPreTrainedModel, Layers.LayerNorm, cfg, state_dict, joinname(lprefix, "input_layernorm"))
        sa = load_model(HGFLlamaPreTrainedModel, SelfAttention, cfg, state_dict, joinname(lprefix, "self_attn"))
        ff = load_model(HGFLlamaPreTrainedModel, Layers.Chain{Tuple{Layers.Dense, Layers.Dense}}, cfg, state_dict, joinname(lprefix, "mlp"))
        # sa = Layers.PreNormResidual(sa, 

        # block = TransformerBlock(sa, ff)
        # push!(blocks, block)
    end
    collect_f = collect_output ? Layers.collect_outputs : nothing
    trf = Transformer(Tuple(blocks), collect_f)
    final_ln = load_model(HGFPhiPreTrainedModel, Layers.LayerNorm, cfg, state_dict, joinname(prefix, "norm"))
    return Layers.Chain(trf, final_ln)
end

# function load_model(_type::Type{<:HGFLlamaPreTrainedModel}, ::Type{<:CompositeEmbedding}, cfg, state_dict, prefix)
# function load_model(    _type::Type{<:HGFLlamaPreTrainedModel}, ::Type{<:Layers.Chain{<:Tuple{Layers.Dense, Layers.Dense}}}, cfg, state_dict, prefix)
