import ..Layers

using Flux
using NNlib
using Tricks
using StructWalk
using DataStructures: OrderedDict

using LinearAlgebra

struct FirstTokenPooler end
(m::FirstTokenPooler)(x) = selectdim(x, 2, 1)

abstract type HGFPreTrainedModel end
Layers.@fluxshow HGFPreTrainedModel

abstract type HGFPreTrained{model_type, task} <: HGFPreTrainedModel end

getmodeltype(m::HGFPreTrained) = getmodeltype(typeof(m))
getmodeltask(m::HGFPreTrained) = getmodeltask(typeof(m))
getmodeltype(::Type{<:HGFPreTrained{MT}}) where MT = MT
getmodeltask(::Type{<:HGFPreTrained{MT, T}}) where {MT, T} = T

function hgf_model_forward end
function hgf_model_loss end

function (model::HGFPreTrained)(nt::NamedTuple)
    if static_hasmethod(hgf_model_loss, Tuple{typeof(model)}) && haskey(nt, :label)
        return hgf_model_loss(model)(model, hgf_model_forward(model, nt))
    else
        return hgf_model_forward(model, nt)
    end
end

for (task, lfunc) in (
    (:forcausallm, :causal_lm_loss),
)
    @eval begin
        @inline hgf_model_loss(::HGFPreTrained{MT, $(QuoteNode(task))}) where MT = $lfunc
    end
end

function _hgfmodelstruct(model_type, type_name, task_name, field_names, expr = nothing)
    sbody = []
    tnames = []
    fbody = :nt
    for fname in field_names
        tname = Symbol(uppercase(String(fname)))
        push!(tnames, tname)
        push!(sbody, :($fname::$tname))
        fbody = :(model.$fname($fbody))
    end
    sname = Symbol("HGF", type_name, task_name)
    name = Expr(:<:,
                Expr(:curly, sname, tnames...),
                Expr(:curly, :HGFPreTrained, QuoteNode(model_type), QuoteNode(Symbol(lowercase(String(task_name))))))
    st = Expr(:struct, false, name, Expr(:block, sbody...))
    if !isnothing(expr)
        fbody = expr.args
    else
        fbody = (fbody,)
    end
    func = :(@inline $(@__MODULE__).hgf_model_forward(model::$sname, nt::NamedTuple) = $(fbody...))
    return Expr(:block, st, :($(@__MODULE__).@functor $sname), func)
end

function _extractfields(ex)
    field_names = Symbol[]
    function __extractfields(x)
        if Meta.isexpr(x, :call)
            f = x.args[1]
            Meta.isexpr(f, :.) && f.args[1] == :model &&
                push!(field_names, f.args[2].value)
        end
        return nothing
    end
    StructWalk.scan(__extractfields, identity, __extractfields, StructWalk.WalkStyle, ex)
    return field_names
end

function _modeldef(model_type, type_name, ex)
    ex isa Symbol &&
        return _hgfmodelstruct(model_type, type_name, ex, (:model, :cls))
    if Meta.isexpr(ex, :call, 3) && first(ex.args) == :(=>)
        task_name = ex.args[2]
        ex = ex.args[3]
        if Meta.isexpr(ex, :tuple)
            all(Base.Fix2(isa, Symbol), ex.args) &&
                return _hgfmodelstruct(model_type, type_name, task_name, ex.args)
        elseif Meta.isexpr(ex, :block)
            field_names = _extractfields(ex)
            st = _hgfmodelstruct(model_type, type_name, task_name, field_names, ex)
            return st
        end
    end
    error("Unknown pattern: $ex")
end

macro hgfdef(type_name, ex)
    model_type = QuoteNode(Symbol(lowercase(String(type_name))))
    return var"@hgfdef"(__source__, __module__, model_type, type_name, ex)
end
macro hgfdef(model_type, type_name, ex)
    if model_type isa Symbol
        model_type = QuoteNode(model_type)
    end
    @assert model_type isa QuoteNode "model_type is not a Symbol"
    @assert type_name isa Symbol
    @assert Meta.isexpr(ex, :tuple) "supported models should be put in a tuple"
    exprs = []
    for task in ex.args
        st = _modeldef(model_type.value, type_name, task)
        append!(exprs, st.args)
    end
    return esc(Expr(:block, :(const $(Symbol(:HGF, type_name, :PreTrainedModel)) = HGFPreTrained{$model_type}), exprs...))
end

isbasemodel(_) = false
isbasemodel(::Type{<:HGFPreTrained{T, :model}}) where T = true

const ACT2FN = @alias (
    [gelu, gelu_new, quick_gelu] = gelu,
    [swish, silu] = swish,
    relu = relu,
    mish = mish,
    selu = selu,
)

joinname(prefix, name) = isempty(prefix) ? name : join((prefix, name), '.')
joinname(prefix, n1, n2...) = joinname(prefix, join((n1, n2...), '.'))

haskeystartswith(dict, prefix) = any(startswith("$prefix."), keys(dict))

zero_init(dims) = () -> zeros(Float32, dims)
one_init(dims) = () -> ones(Float32, dims)
function weight_init(din, dout, factor = true)
    function weight_init_f() # normal(mean = 0, std = factor)
        weight = randn(Float32, dout, din)
        if !isone(factor)
            weight .*= factor
        end
        return weight
    end
    return weight_init_f
end

collect32(x) = collect(Float32, x)

getweight(init, ::Type, ::Symbol) = init()
getweight(init,  x, sym::Symbol) = getproperty(x, sym)

getweight(init, ::Type{<:Array}, state_dict, name) = _getweight(collect32, init, state_dict, name)
getweight(init, ::Type{<:Array}, state_dict::OrderedDict{String}, name) = getweight(init, state_dict, name)
getweight(init, ::Type{<:Layers.Embed}, state_dict, name) = _getweight(collect32 ∘ adjoint, init, state_dict, name)
getweight(init, ::Type{<:Layers.Embed}, state_dict::OrderedDict{String}, name) = _getweight(adjoint, init, state_dict, name)

getweight(init, state_dict, name) = _getweight(identity, init, state_dict, name)
function _getweight(process, init, state_dict, name)
    if haskey(state_dict, name)
        state = state_dict[name]
        if Pickle.Torch.islazy(state)
            lazystate = state
            if Pickle.Torch.isloaded(lazystate)
                weight = lazystate.data
            else
                state = lazystate()
                weight = process(state)
                lazystate.data = weight
            end
        else
            weight = process(state)
        end
    else
        @debug "$name not found, initialized."
        weight = init()
    end
    return weight
end

get_state_dict(_, m::Layers.Embed, state_dict, prefix) = get_state_dict(m, state_dict, prefix)
function get_state_dict(m::Layers.Embed, state_dict, prefix)
    state_dict[joinname(prefix, "weight")] = m.embeddings'
    return state_dict
end

get_state_dict(_, m::Layers.EmbedDecoder, state_dict, prefix) = get_state_dict(m, state_dict, prefix)
function get_state_dict(m::Layers.EmbedDecoder, state_dict, prefix)
    if !isnothing(m.bias)
        state_dict[joinname(prefix, "bias")] = m.bias
    end
    get_state_dict(m.embed, state_dict, prefix)
    return state_dict
end

get_state_dict(_, m::Layers.FixedLenPositionEmbed, state_dict, prefix) = get_state_dict(m, state_dict, prefix)
function get_state_dict(m::Layers.FixedLenPositionEmbed, state_dict, prefix)
    state_dict[joinname(prefix, "weight")] = m.embeddings'
    return state_dict
end

get_state_dict(_, m::Layers.Dense, state_dict, prefix) = get_state_dict(m, state_dict, prefix)
function get_state_dict(m::Layers.Dense, state_dict, prefix)
    state_dict[joinname(prefix, "weight")] = m.W
    !isnothing(m.b) && (state_dict[joinname(prefix, "bias")] = m.b)
    return state_dict
end

get_state_dict(_, m::Layers.LayerNorm, state_dict, prefix) = get_state_dict(m, state_dict, prefix)
function get_state_dict(m::Layers.LayerNorm, state_dict, prefix)
    state_dict[joinname(prefix, "weight")] = m.α
    state_dict[joinname(prefix, "bias")] = m.β
    return state_dict
end

get_state_dict(_, m::Layers.RMSLayerNorm, state_dict, prefix) = get_state_dict(m, state_dict, prefix)
function get_state_dict(m::Layers.RMSLayerNorm, state_dict, prefix)
    state_dict[joinname(prefix, "weight")] = m.α
    return state_dict
end

get_state_dict(p, m::Layers.RenameArgs, state_dict, prefix) = get_state_dict(p, m.layer, state_dict, prefix)
get_state_dict(p, m::Layers.Branch, state_dict, prefix) = get_state_dict(p, m.layer, state_dict, prefix)
get_state_dict(p, m::Layers.Parallel, state_dict, prefix) = get_state_dict(p, m.layer, state_dict, prefix)
get_state_dict(p, m::Layers.DropoutLayer, state_dict, prefix) = get_state_dict(p, m.layer, state_dict, prefix)

load_model(_type::Type, cfg) = load_model(_type, cfg, OrderedDict{String, Any}())
load_model(_type::Type, cfg, state_dict) = load_model(_type, cfg, state_dict, "")

get_state_dict(m) = get_state_dict(m, OrderedDict{String, Any}())
get_state_dict(m, state_dict) = get_state_dict(m, state_dict, "")


_load_embed(state_dict, prefix, vocab_size, dims, factor, pad_idx0 = nothing) =
    _load_embed(state_dict, prefix, weight_init(vocab_size, dims, factor), pad_idx0)
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

function _load_layernorm(state_dict, prefix, dims, ln_ϵ)
    old_weight_name = joinname(prefix, "gamma")
    old_bias_name = joinname(prefix, "beta")
    weight_name = haskey(state_dict, old_weight_name) ? old_weight_name : joinname(prefix, "weight")
    bias_name = haskey(state_dict, old_bias_name) ? old_bias_name : joinname(prefix, "bias")
    ln_weight = getweight(one_init(dims), Array, state_dict, weight_name)
    ln_bias = getweight(zero_init(dims), Array, state_dict, bias_name)
    return Layers.LayerNorm(ln_weight, ln_bias, ln_ϵ)
end

_load_dense(state_dict, prefix, din, dout, factor, bias, act = nothing) =
    _load_dense(state_dict, prefix, weight_init(din, dout, factor), bias ? zero_init(dout) : nothing, act)
function _load_dense(state_dict, prefix, w_init, b_init, act = nothing)
    weight = getweight(w_init, Array, state_dict, joinname(prefix, "weight"))
    if isnothing(b_init)
        bias = nothing
    else
        bias = getweight(b_init, Array, state_dict, joinname(prefix, "bias"))
    end
    return Layers.Dense(act, weight, bias)
end
