using ..Layers
using ..Layers: CompositeEmbedding, SelfAttention
using ChainRulesCore
using Functors
using Static

using NeuralAttentionlib
using NeuralAttentionlib
using NeuralAttentionlib: $, AbstractAttenOp, WithScore, BatchedMask, CausalMask, LocalMask,
    dot_product_score, masked_score, normalized_score, dropout_score, weighted_sum_mixing,
    generic_multihead_qkv_attention

dot_attention(mask, p) =
    dropout_score(p) $ normalized_score(softmax) $
    masked_score(NeuralAttentionlib.GenericMaskOp(), mask) $ dot_product_score

ChainRulesCore.@non_differentiable dot_attention(arg...)

function multihead_qkv_dot_attention(head, q, k, v, mask = nothing, p = nothing)
    return generic_multihead_qkv_attention(weighted_sum_mixing, dot_attention(mask, p), head, q, k, v)
end
function multihead_qkv_dot_attention(
    ::typeof(NeuralAttentionlib.score_returning),
    head, q, k, v, mask = nothing, p = nothing)
    return generic_multihead_qkv_attention(
        NeuralAttentionlib.score_returning(weighted_sum_mixing),
        dot_attention(mask, p), head, q, k, v)
end

struct CausalMultiheadQKVDotAttenOp{F} <: AbstractAttenOp
    head::Int
    p::F
end
CausalMultiheadQKVDotAttenOp(head) = CausalMultiheadQKVDotAttenOp(head, nothing)
NeuralAttentionlib.get_attention_func(::CausalMultiheadQKVDotAttenOp) = multihead_qkv_dot_attention
NeuralAttentionlib.get_attention_func_args(op::CausalMultiheadQKVDotAttenOp, q, k, v, mask = nothing) =
    (op.head, q, k, v, BatchedMask(CausalMask() & mask), op.p)

Layers.no_dropout(op::CausalMultiheadQKVDotAttenOp) = CausalMultiheadQKVDotAttenOp(op.head, nothing)

const CausalMultiheadQKVDotAttenOpWithScore{F} = WithScore{CausalMultiheadQKVDotAttenOp{F}}

struct LocalCausalMultiheadQKVDotAttenOp{F} <: AbstractAttenOp
    size::Int
    head::Int
    p::F
end
LocalCausalMultiheadQKVDotAttenOp(size, head) = LocalCausalMultiheadQKVDotAttenOp(size, head, nothing)
NeuralAttentionlib.get_attention_func(::LocalCausalMultiheadQKVDotAttenOp) = multihead_qkv_dot_attention
NeuralAttentionlib.get_attention_func_args(op::LocalCausalMultiheadQKVDotAttenOp, q, k, v, mask = nothing) =
    (op.head, q, k, v, BatchedMask(CausalMask() & LocalMask(op.size) & mask), op.p)

Layers.no_dropout(op::LocalCausalMultiheadQKVDotAttenOp) = LocalCausalMultiheadQKVDotAttenOp(op.size, op.head, nothing)

const LocalCausalMultiheadQKVDotAttenOpWithScore{F} = WithScore{LocalCausalMultiheadQKVDotAttenOp{F}}

Layers.argument_names(
    ::Union{CausalMultiheadQKVDotAttenOpWithScore, CausalMultiheadQKVDotAttenOp,
            LocalCausalMultiheadQKVDotAttenOpWithScore, LocalCausalMultiheadQKVDotAttenOp}
) = (:hidden_state, :attention_mask)

function Layers.apply_on_namedtuple(
    op::Union{
        CausalMultiheadQKVDotAttenOpWithScore, CausalMultiheadQKVDotAttenOp,
        LocalCausalMultiheadQKVDotAttenOpWithScore, LocalCausalMultiheadQKVDotAttenOp},
    nt::NamedTuple
)
    return Layers.apply_attention_op(op, nt)
end

abstract type HGFGPTNeoPreTrainedModel <: HGFPreTrainedModel end

struct HGFGPTNeoModel{E, DEC} <: HGFGPTNeoPreTrainedModel
    embed::E
    decoder::DEC
end
@functor HGFGPTNeoModel

(model::HGFGPTNeoModel)(nt::NamedTuple) = model.decoder(model.embed(nt))

for T in :[
    HGFGPTNeoForCausalLM
].args
    @eval begin
        struct $T{M, C} <: HGFGPTNeoPreTrainedModel
            model::M
            cls::C
        end
        @functor $T
        (model::$T)(nt::NamedTuple) = model.cls(model.model(nt))
    end
end

for T in :[
    HGFGPTNeoModel, HGFGPTNeoForCausalLM
].args
    @eval function Base.show(io::IO, m::MIME"text/plain", x::$T)
        if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
            Flux._big_show(io, x)
        elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
            Flux._layer_show(io, x)
        else
            show(io, x)
        end
    end
end

basemodelkey(::Type{<:HGFGPTNeoPreTrainedModel}) = :transformer
isbasemodel(::Type{<:HGFGPTNeoModel}) = true
isbasemodel(::Type{<:HGFGPTNeoPreTrainedModel}) = false

get_model_type(::Val{:gpt_neo}) = (
    model = HGFGPTNeoModel,
    forcausallm = HGFGPTNeoForCausalLM,
)

function load_model(_type::Type{HGFGPTNeoModel}, cfg, state_dict, prefix)
    embed = load_model(_type, CompositeEmbedding, cfg, state_dict, prefix)
    decoder = load_model(_type, TransformerBlock, cfg, state_dict, prefix)
    return HGFGPTNeoModel(embed, decoder)
end

function load_model(_type::Type{HGFGPTNeoForCausalLM}, cfg, state_dict, prefix)
    model = load_model(HGFGPTNeoModel, cfg, state_dict, joinname(prefix, "transformer"))
    if cfg[:tie_word_embeddings]
        embedding = model.embed.layer.token.embeddings
    else
        vocab_size, dims, factor = cfg[:vocab_size], cfg[:hidden_size], Float32(cfg[:initializer_range])
        embedding = getweight(weight_init(vocab_size, dims, factor), Layers.Embed,
                              state_dict, joinname(prefix, "lm_head.weight"))
    end
    lmhead = Layers.EmbedDecoder(Layers.Embed(embedding))
    return HGFGPTNeoForCausalLM(model, Layers.Branch{(:logit,), (:hidden_state,)}(lmhead))
end

function load_model(_type::Type{<:HGFGPTNeoPreTrainedModel}, ::Type{<:CompositeEmbedding}, cfg, state_dict, prefix)
    vocab_size, dims, max_pos = cfg[:vocab_size], cfg[:hidden_size], cfg[:max_position_embeddings]
    factor = Float32(cfg[:initializer_range])
    token_weight = getweight(weight_init(vocab_size, dims, factor), Layers.Embed, state_dict, joinname(prefix, "wte.weight"))
    p = cfg[:embed_dropout]; p = iszero(p) ? nothing : p
    pos_weight = getweight(weight_init(max_pos, dims, factor), Layers.Embed, state_dict, joinname(prefix, "wpe.weight"))
    embed = CompositeEmbedding(
        token = Layers.Embed(token_weight),
        position = Layers.FixedLenPositionEmbed(pos_weight)
    )
    return Layers.DropoutLayer(embed, p)
end

function load_model(_type::Type{<:HGFGPTNeoPreTrainedModel}, ::Type{<:Layers.LayerNorm}, cfg, state_dict, prefix)
    dims = cfg[:hidden_size]
    ln_ϵ = Float32(cfg[:layer_norm_epsilon])
    old_weight_name = joinname(prefix, "gamma")
    old_bias_name = joinname(prefix, "beta")
    weight_name = haskey(state_dict, old_weight_name) ? old_weight_name : joinname(prefix, "weight")
    bias_name = haskey(state_dict, old_bias_name) ? old_bias_name : joinname(prefix, "bias")
    ln_weight = getweight(() -> ones(Float32, dims), Array, state_dict, weight_name)
    ln_bias = getweight(() -> zeros(Float32, dims), Array, state_dict, bias_name)
    return Layers.LayerNorm(ln_weight, ln_bias, ln_ϵ)
end

function load_model(
    _type::Type{<:HGFGPTNeoPreTrainedModel}, ::Type{<:SelfAttention{A}}, cfg, state_dict, prefix
) where A <: Union{CausalMultiheadQKVDotAttenOp, LocalCausalMultiheadQKVDotAttenOp}
    head, dims = cfg[:num_heads], cfg[:hidden_size]
    @assert dims % head == 0 "The hidden size is not a multiple of the number of attention heads."
    p = cfg[:attention_dropout]; p = iszero(p) ? nothing : p
    return_score = cfg[:output_attentions]
    factor = Float32(cfg[:initializer_range])
    q_weight = getweight(weight_init(dims, dims), Array, state_dict, joinname(prefix, "q_proj.weight"))
    k_weight = getweight(weight_init(dims, dims), Array, state_dict, joinname(prefix, "k_proj.weight"))
    v_weight = getweight(weight_init(dims, dims), Array, state_dict, joinname(prefix, "v_proj.weight"))
    qkv_proj = Layers.Fork(Layers.Dense(q_weight), Layers.Dense(k_weight), Layers.Dense(v_weight))
    o_weight = getweight(weight_init(dims, dims, factor), Array, state_dict, joinname(prefix, "out_proj.weight"))
    o_bias = getweight(zero_init(dims), Array, state_dict, joinname(prefix, "out_proj.bias"))
    o_proj = Layers.Dense(o_weight, o_bias)
    if A <: LocalCausalMultiheadQKVDotAttenOp
        window_size = cfg[:window_size]
        op = LocalCausalMultiheadQKVDotAttenOp(window_size, head, p)
    else
        op = CausalMultiheadQKVDotAttenOp(head, p)
    end
    return_score && (op = WithScore(op))
    return SelfAttention(op, qkv_proj, o_proj)
end

function load_model(
    _type::Type{<:HGFGPTNeoPreTrainedModel}, ::Type{<:Layers.Chain{<:Tuple{Layers.Dense, Layers.Dense}}},
    cfg, state_dict, prefix
)
    dims = cfg[:hidden_size]
    ff_dims = get(cfg, :n_inner, nothing)
    ff_dims = isnothing(ff_dims) ? 4dims : ff_dims
    factor = Float32(cfg[:initializer_range])
    act = ACT2FN[Symbol(cfg[:activation_function])]
    wi_weight = getweight(weight_init(dims, ff_dims, factor), Array,
                          state_dict, joinname(prefix, "c_fc.weight"))
    wi_bias = getweight(zero_init(ff_dims), Array, state_dict, joinname(prefix, "c_fc.bias"))
    wo_weight = getweight(weight_init(ff_dims, dims, factor), Array,
                          state_dict, joinname(prefix, "c_proj.weight"))
    wo_bias = getweight(zero_init(dims), Array, state_dict, joinname(prefix, "c_proj.bias"))
    return Layers.Chain(Layers.Dense(act, wi_weight, wi_bias), Layers.Dense(wo_weight, wo_bias))
end

function load_model(_type::Type{<:HGFGPTNeoPreTrainedModel}, ::Type{<:TransformerBlock}, cfg, state_dict, prefix)
    n = cfg[:num_layers]
    p = cfg[:resid_dropout]; p = iszero(p) ? nothing : p
    collect_output = cfg[:output_attentions] || cfg[:output_hidden_states]
    atten_ops = cfg[:attention_layers]
    blocks = []
    for i = 1:n
        lprefix = joinname(prefix, :h, i-1)
        op_str = atten_ops[i]
        op_type = op_str == "global" ?
            CausalMultiheadQKVDotAttenOp : op_str == "local" ?
            LocalCausalMultiheadQKVDotAttenOp : error("unknown attention type: $op_str")
        sa = load_model(_type, SelfAttention{op_type}, cfg, state_dict, joinname(lprefix, "attn.attention"))
        sa_ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(lprefix, "ln_1"))
        sa = Layers.PreNormResidual(Layers.DropoutLayer(sa, p), sa_ln)
        ff = load_model(_type, Layers.Chain{Tuple{Layers.Dense, Layers.Dense}}, cfg, state_dict, joinname(lprefix, "mlp"))
        ff_ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(lprefix, "ln_2"))
        ff = Layers.PreNormResidual(Layers.DropoutLayer(ff, p), ff_ln)
        block = TransformerBlock(sa, ff)
        push!(blocks, block)
    end
    collect_f = collect_output ? Layers.collect_outputs : nothing
    trf = Transformer(Tuple(blocks), collect_f)
    final_ln = load_model(_type, Layers.LayerNorm, cfg, state_dict, joinname(prefix, "ln_f"))
    return Layers.Chain(trf, final_ln)
end
