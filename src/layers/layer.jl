using Functors
using Static

using NeuralAttentionlib
using NeuralAttentionlib: AbstractAttenOp, MultiheadQKVAttenOp, CausalMultiheadQKVAttenOp, dropout

struct DropoutLayer{L, P} <: LayerStruct
    layer::L
    p::P
end
@functor DropoutLayer (layer,)

argument_names(dp::DropoutLayer) = argument_names(dp.layer)

function (dp::DropoutLayer{L, Nothing})(nt::NamedTuple) where L
    y = apply_on_namedtuple(dp.layer, nt)
    return y
end
function (dp::DropoutLayer)(nt::NamedTuple)
    y = apply_on_namedtuple(dp.layer, nt)
    hidden_state = dropout(y.hidden_state, dp.p)
    return return_hidden_state(y, hidden_state)
end

_show_name(dp::DropoutLayer) = join(("DropoutLayer<", dp.p, ">"))

function Base.show(io::IO, dp::DropoutLayer)
    print(io, _show_name(dp))
    print(io, '(')
    show(io, dp.layer)
    print(io, ')')
end

#############################################

abstract type AbstractTransformerBlock <: LayerStruct end

struct TransformerBlock{A, F} <: AbstractTransformerBlock
    attention::A
    feedforward::F
end
@functor TransformerBlock

(b::TransformerBlock)(nt::NamedTuple) = apply_on_namedtuple(b.feedforward, apply_on_namedtuple(b.attention, nt))

struct TransformerDecoderBlock{A, F} <: AbstractTransformerBlock
    attention::A
    feedforward::F
end
@functor TransformerDecoderBlock

argument_names(b::TransformerDecoderBlock) = Base.merge_names(
    argument_names(b.attention),
    argument_names(b.feedforward)
)

# performs attention on nt, returns the result as an NamedTuple
# then performs crossattention on the result, returns the result as an NamedTuple
# then performs feedforward on the result, returns the result as an NamedTuple
# (b::TransformerDecoderBlock)(nt::NamedTuple) =
#     apply_on_namedtuple(b.feedforward, apply_on_namedtuple(b.crossattention, apply_on_namedtuple(b.attention, nt)))
(b::TransformerDecoderBlock)(nt::NamedTuple) =
    apply_on_namedtuple(b.feedforward, apply_on_namedtuple(b.attention, nt))

struct Residual{L} <: LayerStruct
    layer::L
end
@functor Residual

function (resi::Residual)(nt::NamedTuple)
    y = apply_on_namedtuple(resi.layer, nt)
    hidden_state = y.hidden_state + nt.hidden_state
    return return_hidden_state(y, hidden_state)
end

struct PreNormResidual{L, N} <: LayerStruct
    layer::L
    norm::N
end
@functor PreNormResidual

function (prenr::PreNormResidual)(nt::NamedTuple)
    norm = apply_on_namedtuple(prenr.norm, nt)
    y = apply_on_namedtuple(prenr.layer, norm)
    hidden_state = y.hidden_state + nt.hidden_state
    return return_hidden_state(y, hidden_state)
end

struct PostNormResidual{L, N} <: LayerStruct
    layer::L
    norm::N
end
@functor PostNormResidual

function (postnr::PostNormResidual)(nt::NamedTuple)
    y = apply_on_namedtuple(postnr.layer, nt)
    hidden_state = y.hidden_state + nt.hidden_state
    r = return_hidden_state(y, hidden_state)
    return apply_on_namedtuple(postnr.norm, r)
end

const  PreNormTransformerBlock{A, LN1, F, LN2} = TransformerBlock{ PreNormResidual{A, LN1},  PreNormResidual{F, LN2}}
const PostNormTransformerBlock{A, LN1, F, LN2} = TransformerBlock{PostNormResidual{A, LN1}, PostNormResidual{F, LN2}}
const  PreNormTransformerDecoderBlock{A, LN1, #=C, LN2,=# F, LN3} =
    TransformerDecoderBlock{ PreNormResidual{A, LN1},  #=PreNormResidual{C, LN2},=#  PreNormResidual{F, LN3}}
const PostNormTransformerDecoderBlock{A, LN1, #=C, LN2,=# F, LN3} =
    TransformerDecoderBlock{PostNormResidual{A, LN1}, #=PostNormResidual{C, LN2},=# PostNormResidual{F, LN3}}

function Base.show(io::IO, t::PreNormTransformerBlock)
    print(io, "PreNormTransformerBlock(");
    show(io, t.attention.layer); print(io, ", "); show(io, t.attention.norm); print(io, ", ")
    show(io, t.feedforward.layer); print(io, ", "); show(io, t.feedforward.norm); print(io, ')')
end
function Base.show(io::IO, t::PostNormTransformerBlock)
    print(io, "PostNormTransformerBlock(")
    show(io, t.attention.layer); print(io, ", "); show(io, t.attention.norm); print(io, ", ")
    show(io, t.feedforward.layer); print(io, ", "); show(io, t.feedforward.norm); print(io, ')')
end
function Base.show(io::IO, t::PreNormTransformerDecoderBlock)
    print(io, "PreNormTransformerDecoderBlock(")
    show(io, t.attention.layer); print(io, ", "); show(io, t.attention.norm); print(io, ", ");
    # show(io, t.crossattention.layer); print(io, ", "); show(io, t.crossattention.norm); print(io, ", ");
    show(io, t.feedforward.layer); print(io, ", "); show(io, t.feedforward.norm); print(io, ')')
end
function Base.show(io::IO, t::PostNormTransformerDecoderBlock)
    print(io, "PostNormTransformerDecoderBlock(")
    show(io, t.attention.layer); print(io, ", "); show(io, t.attention.norm); print(io, ", ");
    #show(io, t.crossattention.layer); print(io, ", "); show(io, t.crossattention.norm); print(io, ", ");
    show(io, t.feedforward.layer); print(io, ", "); show(io, t.feedforward.norm); print(io, ')')
end
_show_name(t::PreNormTransformerBlock) = "PreNormTransformerBlock"
_show_name(t::PostNormTransformerBlock) = "PostNormTransformerBlock"
_show_name(t::PreNormTransformerDecoderBlock) = "PreNormTransformerDecoderBlock"
_show_name(t::PostNormTransformerDecoderBlock) = "PostNormTransformerDecoderBlock"

Flux._show_children(t::PreNormTransformerBlock) = (t.attention.layer, t.attention.norm, t.feedforward.layer, t.feedforward.norm)
Flux._show_children(t::PostNormTransformerBlock) = (t.attention.layer, t.attention.norm, t.feedforward.layer, t.feedforward.norm)
Flux._show_children(t::PreNormTransformerDecoderBlock) = (t.attention.layer, t.attention.norm, #=t.crossattention.layer, t.crossattention.norm,=# t.feedforward.layer, t.feedforward.norm)
Flux._show_children(t::PostNormTransformerDecoderBlock) = (t.attention.layer, t.attention.norm, #= t.crossattention.layer, t.crossattention.norm, =# t.feedforward.layer, t.feedforward.norm)

#############################################

struct SelfAttention{A, QKV, O} <: LayerStruct
    attention_op::A
    qkv_proj::QKV #::NSplit{StaticInt{3}, QKV}
    o_proj::O
end
@functor SelfAttention

function (sa::SelfAttention)(nt::NamedTuple)
    qkv = apply_on_namedtuple(sa.qkv_proj, nt)
    a = apply_on_namedtuple(sa.attention_op, qkv)
    y = apply_on_namedtuple(sa.o_proj, a)
    # return y 
    # NOTE: instead of returning y, we return a copy of y, because 
    # there is some sort of memory leak when using distributed for Jevo specifically,
    # I suspect related to gradients. This cuts off gradient flow.
    hidden_state = zeros(Float32, size(y.hidden_state)) |> Flux.gpu
    hidden_state .= y.hidden_state 
    return (hidden_state = hidden_state, attention_mask = y.attention_mask)
end

struct CrossAttention{A, Q, KV, O} <: LayerStruct
    attention_op::A
    q_proj::Q
    kv_proj::KV #::NSplit{StaticInt{2}, KV}
    o_proj::O
end
@functor CrossAttention

function argument_names(ca::CrossAttention)
    required_names = (:hidden_state, :memory)
    field_names = invoke(argument_names, Tuple{LayerStruct}, ca)
    cross_field_names = remove_name(prefix_name(:cross, field_names), :cross_hidden_state)
    return Base.merge_names(required_names, cross_field_names)
end

function _apply_cross_attention_op(op, q::NamedTuple, kv::NamedTuple, cross_attention_mask)
    k_v = kv.hidden_state
    ChainRulesCore.ignore_derivatives() do
        k_v isa NTuple{2, Any} ||
            error("Expect kv_proj(memory).hidden_state return a tuple of 2 arrays, but get $(typeof(kv.hidden_state)).")
        nothing
    end
    k, v = k_v
    qkv = merge(kv, q, (
        hidden_state = (q.hidden_state, k, v),
        attention_mask = cross_attention_mask,
    ))
    a = apply_on_namedtuple(op, Base.structdiff(qkv, NamedTuple{(:attention_score,)}))
    return a
end

function (ca::CrossAttention)(nt::NamedTuple)
    hidden_state, memory = nt.hidden_state, nt.memory
    cross_attention_mask = ChainRulesCore.ignore_derivatives(()->get(nt, :cross_attention_mask, nothing))
    nt_ext = Base.structdiff(nt, NamedTuple{(:hidden_state, :memory, :attention_mask, :cross_attention_mask)})
    q = with_extra(ca.q_proj, hidden_state, nt_ext)
    kv = with_extra(ca.kv_proj, memory, nt_ext)
    _a = _apply_cross_attention_op(ca.attention_op, q, kv, cross_attention_mask)
    a = rename(Base.structdiff(_a, NamedTuple{(:attention_mask, :cross_attention_mask)}),
               Val(:attention_score), Val(:cross_attention_score))
    y = apply_on_namedtuple(ca.o_proj, a)
    return merge(nt, y)
end

#############################################

struct Transformer{T <: Tuple{Vararg{<:AbstractTransformerBlock}}, F} <: LayerStruct
    blocks::T
    f::F
end
Transformer(blocks::Tuple{Vararg{AbstractTransformerBlock}}) = Transformer(blocks, nothing)
Transformer(blocks::AbstractTransformerBlock...) = Transformer(blocks)

@functor Transformer

(t::Transformer)(nt::NamedTuple) = applyblocks(t.blocks, t.f, nt)

function _block_call(symbols, i, has_f)
    call = :(blocks[$i]($(symbols[i])))
    if has_f
        call = :(f($(symbols[i]), $call))
    end
    line = :($(symbols[i+1]) = $call)
    return line
end

function applyblocks(blocks::Tuple{Vararg{AbstractTransformerBlock, N}}, f, x) where N
    if @generated
        symbols = vcat(:x, [gensym() for _ in 1:N])
        has_f = !(f <: Nothing)
        calls = [ _block_call(symbols, i, has_f) for i in 1:N ]
        return Expr(:block, calls...)
    else
        if isnothing(f)
            return foldl((y, blk)-> blk(y), blocks; init=x)
        else
            return foldl((y, blk)-> f(y, blk(y)), blocks; init=x)
        end
    end
end

Base.getindex(t::Transformer, i::Integer) = t.blocks[i]
Base.getindex(t::Transformer, r::AbstractVector) = Transformer(t.blocks[r])
Base.length(t::Transformer) = length(t.blocks)

"""
    Transformer(T::Type{<:AbstractTransformerBlock}, n::Int, args...; kwargs...)

Create `n` layers of transformer blocks with `T(args...; kwargs...)`.
"""
function Transformer(T::Type{<:AbstractTransformerBlock}, n::Int, args...; collect_outputs = false, kwargs...)
    collect_f = collect_outputs isa Bool ?
        (collect_outputs ? (@__MODULE__).collect_outputs : nothing) :
        collect_outputs
    return Transformer(ntuple(i -> T(args...; kwargs...), n), collect_f)
end

function Base.show(io::IO, t::Transformer)
    print(io, "Transformer")
    ts = t.blocks
    if ts isa NTuple
        N = length(ts)
        print(io, "<$N>(")
        show(io, first(ts))
        print(io, ')')
    else
        show(io, ts)
    end
end
function Flux._big_show(io::IO, t::Transformer, indent::Int = 0, name = nothing)
    if t.blocks isa NTuple
        println(io, " "^indent, isnothing(name) ? "" : "$name = ", _show_name(t), "<$(length(t))>(")
        Flux._big_show(io, first(t.blocks), indent + 2)
    else
        println(io, " "^indent, isnothing(name) ? "" : "$name = ", _show_name(t), '(')
        for c in t.blocks
            Flux._big_show(io, c, indent + 2)
        end
    end
    if iszero(indent)
        print(io, rpad(')', 2))
        Flux._big_finale(io, t)
    else
        print(io, " "^indent, "),")
        Flux._big_finale(io, t)
        print(io, '\n')
    end
end

#############################################

function collect_outputs(prev::NamedTuple{prev_names, types}, output::NamedTuple{names}) where {prev_names, types, names}
    if @generated
        if iszero(sym_in(:outputs, names))
            return quote
                new_output = Base.structdiff(output, prev)
                outputs = (merge((hidden_state = output.hidden_state,), new_output),)
                return merge(output, (outputs = outputs,))
            end
        else
            i = sym_in(:outputs, prev_names)
            name = types.parameters[i].parameters[1].parameters[1]
            return quote
                prev_outputs = prev.outputs
                new_output = NamedTuple{$name}(output)
                outputs = (prev_outputs..., new_output)
                return merge(output, (outputs = outputs,))
            end
        end
    else
        if haskey(prev, :outputs)
            prev_outputs = prev.outputs
            new_output = NamedTuple{keys(first(prev_outputs))}(output) # assume each block give the same outputs
            outputs = (prev_outputs..., new_output)
        else
            new_output = Base.structdiff(output, prev)
            outputs = (merge((hidden_state = output.hidden_state,), new_output),)
        end
        return merge(output, (outputs = outputs,))
    end
end

#############################################

"""
    SelfAttention(head::Int, hidden_size::Int [, head_hidden_size::Int = hidden_size ÷ head ];
                  dropout::Union{Nothing, Float64} = nothing, return_score = false, causal = false)

Create a multi-head self attention layer with `head` heads and `head_hidden_size` per head.
"""
function SelfAttention(head::Int, hidden_size::Int; dropout = nothing, return_score = false, causal = false)
    @assert rem(hidden_size, head) == 0 "`hidden_size` should be dividible by `head` if `head_hidden_size` is not set"
    head_hidden_size = div(hidden_size, head)
    return SelfAttention(head, hidden_size, head_hidden_size; dropout, return_score, causal)
end
function SelfAttention(
    head::Int, hidden_size::Int, head_hidden_size::Int;
    dropout = nothing, return_score = false, causal = false,
)
    atten_op_constr = causal ? CausalMultiheadQKVAttenOp : MultiheadQKVAttenOp
    atten_op = atten_op_constr(head, dropout)
    return_score && (atten_op = NeuralAttentionlib.WithScore(atten_op))
    sa = SelfAttention(atten_op, head, hidden_size, head_hidden_size)
    return sa
end

"""
    SelfAttention(atten_op::AbstractAttenOp, head::Int, hidden_size::Int, head_hidden_size::Int)

Create a self attention layer with given `atten_op`.
"""
function SelfAttention(atten_op::AbstractAttenOp, head::Int, hidden_size::Int, head_hidden_size::Int)
    qkv_proj = Dense(hidden_size, 3head * head_hidden_size)
    o_proj = Dense(head * head_hidden_size, hidden_size)
    sa = SelfAttention(atten_op, NSplit(static(3), qkv_proj), o_proj)
    return sa
end

"""
    CrossAttention(head::Int, hidden_size::Int [, head_hidden_size::Int = hidden_size ÷ head ];
                   dropout::Union{Nothing, Float64} = nothing, return_score = false)

Create a multi-head cross attention layer with `head` heads and `head_hidden_size` per head.
"""
function CrossAttention(head::Int, hidden_size::Int; dropout = nothing, return_score = false)
    @assert rem(hidden_size, head) == 0 "`hidden_size` should be dividible by `head` if `head_hidden_size` is not set"
    head_hidden_size = div(hidden_size, head)
    return CrossAttention(head, hidden_size, head_hidden_size; dropout, return_score)
end
function CrossAttention(head::Int, hidden_size::Int, head_hidden_size::Int; dropout = nothing, return_score = false)
    cross_atten_op = MultiheadQKVAttenOp(head, dropout)
    return_score && (cross_atten_op = NeuralAttentionlib.WithScore(cross_atten_op))
    ca = CrossAttention(cross_atten_op, head, hidden_size, head_hidden_size)
    return ca
end

"""
    CrossAttention(atten_op::AbstractAttenOp, head::Int, hidden_size::Int, head_hidden_size::Int)

Create a cross attention layer with given `atten_op`.
"""
function CrossAttention(cross_atten_op::AbstractAttenOp, head::Int, hidden_size::Int, head_hidden_size::Int)
    c_q_proj = Dense(hidden_size, head * head_hidden_size)
    c_kv_proj = Dense(hidden_size, 2head * head_hidden_size)
    c_o_proj = Dense(head * head_hidden_size, hidden_size)
    ca = CrossAttention(cross_atten_op, c_q_proj, NSplit(static(2), c_kv_proj), c_o_proj)
    return ca
end

#############################################

"""
    TransformerBlock([act,] head::Int, hidden_size::Int [, head_hidden_size::Int], intermediate_size::Int;
                     attention_dropout = nothing, dropout = nothing, return_score = false)

Create a post-LN transformer encoder block. `head`, `hidden_size` (and `head_hidden_size`) are parameters of
 [`SelfAttention`](@ref). `intermediate_size`, `hidden_size` (and `act`) would be use to create the 2 layered
 feed-forward layer.
"""
TransformerBlock(
    head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    attention_dropout = nothing, dropout = nothing, return_score = false,
) = TransformerBlock(
    gelu, head, hidden_size, head_hidden_size, intermediate_size; attention_dropout, dropout, return_score)

TransformerBlock(
    act, head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    attention_dropout = nothing, dropout = nothing, return_score = false,
) = PostNormTransformerBlock(
    act, head, hidden_size, head_hidden_size, intermediate_size; attention_dropout, dropout, return_score)

"""
    PostTransformerBlock([act,] head::Int, hidden_size::Int [, head_hidden_size::Int], intermediate_size::Int;
                         attention_dropout = nothing, dropout = nothing, return_score = false)

Create a post-LN transformer encoder block. `head`, `hidden_size` (and `head_hidden_size`) are parameters of
 [`SelfAttention`](@ref). `intermediate_size`, `hidden_size` (and `act`) would be use to create the 2 layered
 feed-forward layer.
"""
PostNormTransformerBlock(
    head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    attention_dropout = nothing, dropout = nothing, return_score = false,
) = PostNormTransformerBlock(
    gelu, head, hidden_size, head_hidden_size, intermediate_size; attention_dropout, dropout, return_score)

function PostNormTransformerBlock(
    act, head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    attention_dropout = nothing, dropout = nothing, return_score = false,
)
    sa = SelfAttention(head, hidden_size, head_hidden_size; dropout = attention_dropout, return_score)
    ff1 = Dense(act, hidden_size, intermediate_size)
    ff2 = Dense(intermediate_size, hidden_size)
    return TransformerBlock(
        PostNormResidual(
            DropoutLayer(sa, dropout),
            LayerNorm(hidden_size)),
        PostNormResidual(
            DropoutLayer(Chain(ff1, ff2), dropout),
            LayerNorm(hidden_size)))
end

"""
    PreNormTransformerBlock([act,] head::Int, hidden_size::Int [, head_hidden_size::Int], intermediate_size::Int;
                            attention_dropout = nothing, dropout = nothing, return_score = false)

Create a pre-LN transformer encoder block. `head`, `hidden_size` (and `head_hidden_size`) are parameters of
 [`SelfAttention`](@ref). `intermediate_size`, `hidden_size` (and `act`) would be use to create the 2 layered
 feed-forward layer.
"""
PreNormTransformerBlock(
    head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    attention_dropout = nothing, dropout = nothing, return_score = false,
) = PreNormTransformerBlock(
    gelu, head, hidden_size, head_hidden_size, intermediate_size; attention_dropout, dropout, return_score)

function PreNormTransformerBlock(
    act, head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    attention_dropout = nothing, dropout = nothing, return_score = false,
)
    sa = SelfAttention(head, hidden_size, head_hidden_size; dropout = attention_dropout, return_score)
    ff1 = Dense(act, hidden_size, intermediate_size)
    ff2 = Dense(intermediate_size, hidden_size)
    return TransformerBlock(
        PreNormResidual(
            DropoutLayer(sa, dropout),
            LayerNorm(hidden_size)),
        PreNormResidual(
            DropoutLayer(Chain(ff1, ff2), dropout),
            LayerNorm(hidden_size)))
end

#############################################

"""
    TransformerDecoderBlock([act,] head::Int, hidden_size::Int [, head_hidden_size::Int], intermediate_size::Int;
                            attention_dropout = nothing, dropout = nothing, cross_attention_dropout = nothing,
                            return_score = false, return_self_attention_score = false)

Create a post-LN transformer decoder block. `head`, `hidden_size` (and `head_hidden_size`) are parameters of
 [`SelfAttention`](@ref) and [`CrossAttention`](@ref). `intermediate_size`, `hidden_size` (and `act`) would
 be use to create the 2 layered feed-forward layer.
"""
TransformerDecoderBlock(
    head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    dropout = nothing, attention_dropout = nothing, cross_attention_dropout = nothing,
    return_score = false, return_self_attention_score = false,
) = TransformerDecoderBlock(
    gelu, head, hidden_size, head_hidden_size, intermediate_size;
    dropout, attention_dropout, cross_attention_dropout, return_score, return_self_attention_score
)

TransformerDecoderBlock(
    act, head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    dropout = nothing, attention_dropout = nothing, cross_attention_dropout = nothing,
    return_score = false, return_self_attention_score = false,
) = PostNormTransformerDecoderBlock(
    act, head, hidden_size, head_hidden_size, intermediate_size;
    dropout, attention_dropout, cross_attention_dropout, return_score, return_self_attention_score
)

"""
    PostTransformerDecoderBlock([act,] head::Int, hidden_size::Int [, head_hidden_size::Int], intermediate_size::Int;
                                attention_dropout = nothing, dropout = nothing, cross_attention_dropout = nothing,
                                return_score = false, return_self_attention_score = false)

Create a post-LN transformer decoder block. `head`, `hidden_size` (and `head_hidden_size`) are parameters of
 [`SelfAttention`](@ref) and [`CrossAttention`](@ref). `intermediate_size`, `hidden_size` (and `act`) would
 be use to create the 2 layered feed-forward layer.
"""
PostNormTransformerDecoderBlock(
    head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    dropout = nothing, attention_dropout = nothing, cross_attention_dropout = nothing,
    return_score = false, return_self_attention_score = false,
) = PostNormTransformerDecoderBlock(
    gelu, head, hidden_size, head_hidden_size, intermediate_size;
    dropout, attention_dropout, cross_attention_dropout, return_score, return_self_attention_score
)

function PostNormTransformerDecoderBlock(
    act, head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    dropout = nothing, attention_dropout = nothing, cross_attention_dropout = nothing,
    return_score = false, return_self_attention_score = false,
)
    sa = SelfAttention(head, hidden_size, head_hidden_size;
                       dropout = attention_dropout, causal = true, return_score = return_self_attention_score)
    ff1 = Dense(act, hidden_size, intermediate_size)
    ff2 = Dense(intermediate_size, hidden_size)
    return TransformerDecoderBlock(
        PostNormResidual(
            sa,
            LayerNorm(hidden_size)),
        PostNormResidual(
            Chain(ff1, ff2),
            LayerNorm(hidden_size)))
end

"""
    PreTransformerDecoderBlock([act,] head::Int, hidden_size::Int [, head_hidden_size::Int], intermediate_size::Int;
                               attention_dropout = nothing, dropout = nothing, cross_attention_dropout = nothing,
                               return_score = false, return_self_attention_score = false)

Create a pre-LN transformer decoder block. `head`, `hidden_size` (and `head_hidden_size`) are parameters of
 [`SelfAttention`](@ref) and [`CrossAttention`](@ref). `intermediate_size`, `hidden_size` (and `act`) would
 be use to create the 2 layered feed-forward layer.
"""
PreNormTransformerDecoderBlock(
    head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    dropout = nothing, attention_dropout = nothing, cross_attention_dropout = nothing,
    return_score = false, return_self_attention_score = false,
) = PreNormTransformerDecoderBlock(
    gelu, head, hidden_size, head_hidden_size, intermediate_size;
    dropout, attention_dropout, cross_attention_dropout, return_score, return_self_attention_score
)

function PreNormTransformerDecoderBlock(
    act, head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    dropout = nothing, attention_dropout = nothing, cross_attention_dropout = nothing,
    return_score = false, return_self_attention_score = false,
)
    sa = SelfAttention(head, hidden_size, head_hidden_size;
                       dropout = attention_dropout, causal = true, return_score = return_self_attention_score)
    ca = CrossAttention(head, hidden_size, head_hidden_size; dropout = cross_attention_dropout, return_score)
    ff1 = Dense(act, hidden_size, intermediate_size)
    ff2 = Dense(intermediate_size, hidden_size)
    return TransformerDecoderBlock(
        PreNormResidual(
            DropoutLayer(sa, dropout),
            LayerNorm(hidden_size)),
        PreNormResidual(
            DropoutLayer(ca, dropout),
            LayerNorm(hidden_size)),
        PreNormResidual(
            DropoutLayer(Chain(ff1, ff2), dropout),
            LayerNorm(hidden_size)))
end

#############################################

struct Seq2Seq{E, D} <: LayerStruct
    encoder::E
    decoder::D
end
@functor Seq2Seq

argument_names(::Seq2Seq) = (:encoder_input, :decoder_input)

function (seq2seq::Seq2Seq)(nt::NamedTuple)
    enc = apply_on_namedtuple(seq2seq.encoder, nt.encoder_input)
    dec = apply_on_namedtuple(seq2seq.decoder, merge(nt.decoder_input, (memory = enc.hidden_state,)))
    hidden_state = dec.hidden_state
    return merge(Base.structdiff(nt, NamedTuple{(:encoder_input, :decoder_input)}),
                 (hidden_state = hidden_state, encoder_output = enc, decoder_output = dec))
end

#############################################

struct CompositeEmbedding{T<:Tuple}  <: LayerStruct
    embeds::T
end
Functors.functor(::Type{<:CompositeEmbedding}, x) = ((embeds = getfield(x, :embeds),), y -> CompositeEmbedding(y.embeds))

argument_names(ce::CompositeEmbedding) = remove_name(argument_names(getfield(ce, :embeds)), :hidden_state)

CompositeEmbedding(args...) = CompositeEmbedding{typeof(args)}(args)
function CompositeEmbedding(; kwargs...)
    embeds = []
    for (i, name) in enumerate(keys(kwargs))
        embed = kwargs[name]
        if isone(i)
            push!(embeds, WithArg{(name,)}(embed))
        else
            if embed isa ApplyEmbed
                embed = (embed.apply, embed.embed, embed.indices)
            end
            if !(embed isa Tuple)
                embed = (embed,)
            end
            push!(embeds, WithOptArg{(:hidden_state,), (name,)}(ApplyEmbed(embed...)))
        end
    end
    return CompositeEmbedding(Tuple(embeds))
end

function (ce::CompositeEmbedding)(nt::NamedTuple)
    if @generated
        N = length(ce.parameters[1].parameters)
        symbols = [gensym() for _ in 1:N]
        pushfirst!(symbols, :nt)
        calls = [ :($(symbols[i+1]) = apply_on_namedtuple(ce[$i], $(symbols[i]))) for i in 1:N ]
        return Expr(:block, calls...)
    else
        applylayers(getfield(ce, :embeds), nt)
    end
end

Base.length(ce::CompositeEmbedding) = length(getfield(ce, :embeds))
Base.getindex(ce::CompositeEmbedding, i) = getfield(ce, :embeds)[i]

function Base.getproperty(ce::CompositeEmbedding, sym::Symbol)
    names = propertynames(ce)
    i = sym_in(sym, names)
    iszero(i) && error("Unknown embeddding name: $sym\nPossible names: $(propertynames(ce))")
    return getfield(ce, :embeds)[i].layer
end

function Base.propertynames(ce::CompositeEmbedding)
    names = argument_names.(getfield(ce, :embeds))
    return (first(first(names)), last.(Base.tail(names))...)
end

function Base.show(io::IO, ce::CompositeEmbedding)
    print(io, "CompositeEmbedding(")
    ce1 = ce[1]
    print(io, "$(argument_names(ce1)[1]) = ")
    show(io, ce1.layer)
    for e in Base.tail(getfield(ce, :embeds))
        print(io, ", $(argument_names(e)[2]) = ")
        show(io, e.layer)
    end
    print(io, ')')
end
function Flux._big_show(io::IO, ce::CompositeEmbedding, indent::Int = 0, name = nothing)
    println(io, " "^indent, isnothing(name) ? "" : "$name = ", _show_name(ce), '(')
    for ename in propertynames(ce)
        c = getproperty(ce, ename)
        Flux._big_show(io, c, indent + 2, ename)
    end
    if iszero(indent)
        print(io, rpad(')', 2))
        Flux._big_finale(io, ce)
    else
        println(io, " "^indent, "),")
    end
end

#############################################

for T in :[
    DropoutLayer, SelfAttention, CrossAttention,
    PostNormResidual, PreNormResidual, TransformerBlock, TransformerDecoderBlock,
    Transformer, Seq2Seq, CompositeEmbedding,
].args
    if T == :CompositeEmbedding || T == :Transformer || T == :DropoutLayer
        @eval @fluxshow $T false
    else
        @eval @fluxshow $T
    end
end
