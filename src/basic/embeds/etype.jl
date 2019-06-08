abstract type AbstractEmbed{F} end
abstract type AbstractBroadcastEmbed{F} <: AbstractEmbed{F} end

Base.eltype(::AbstractEmbed{F}) where F = F
_getF(::Type{<:AbstractEmbed{F}}) where F = F

struct CompositeEmbedding{F, T <: NamedTuple, T2 <: NamedTuple} <: AbstractEmbed{F}
    embeddings::T
    aggregator::T2
    CompositeEmbedding(es::NamedTuple, as::NamedTuple) = new{_getF(eltype(es)), typeof(es), typeof(as)}(es, as)
end

function CompositeEmbedding(;es...)
    emb = map(x-> x isa Tuple ? x[1] : x, es.data)
    agg = map(x-> x isa Tuple ? x[2] : +, es.data)
    CompositeEmbedding(emb, agg)
end

Flux.children(ce::CompositeEmbedding) = values(ce.embeddings)
Flux.mapchildren(f, ce::CompositeEmbedding) = CompositeEmbedding(;map((e, a)->(f(e), a), ce.embeddings, ce.aggregator)...)

function Base.show(io::IO, e::CompositeEmbedding)
    print(io, "CompositeEmbedding")
    print(io, e.embeddings)
end

get_value(e::AbstractEmbed, name::Symbol, xs::NamedTuple) = e(xs[name])
_repeatdims(bs, vs) = ntuple(i -> i > length(vs) ? bs[i] : 1, length(vs))

@inline aggregate(e::E, f::F, base::A, value::A2) where {F, T, A <: AbstractArray{T}, A2 <: AbstractArray{T}, E <: AbstractEmbed{T}} = f(base, value)
@inline aggregate(e::Eb, f::F, base::A, value::A2) where {F, T, A <: AbstractArray{T}, A2 <: AbstractArray{T}, Eb <: AbstractBroadcastEmbed{T}} = f.(base, value)
@inline aggregate(e::Eb, f::typeof(vcat), base::A, value::A2) where {T, A <: AbstractArray{T}, A2 <: AbstractArray{T}, Eb <: AbstractBroadcastEmbed{T}} = vcat(base, repeat(value, _repeatdims(size(base), size(value))...))
@inline aggregate(e::Eb, f::typeof(hcat), base::A, value::A2) where {T, A <: AbstractArray{T}, A2 <: AbstractArray{T}, Eb <: AbstractBroadcastEmbed{T}} = hcat(base, repeat(value, _repeatdims(size(base), size(value))...))

(ce::CompositeEmbedding)(;xs...) = ce(xs.data)
function (ce::CompositeEmbedding{F})(xs::NamedTuple) where F
    names = keys(ce.embeddings)
    firstname = first(names)
    init = get_value(ce.embeddings[firstname], firstname, xs)

    for name ∈ Base.tail(names)
        emb = ce.embeddings[name]
        agg = ce.aggregator[name]
        value = get_value(emb, name, xs)
        init = aggregate(emb, agg, init, value)
    end

    init
end
