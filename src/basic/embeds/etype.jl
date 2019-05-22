abstract type AbstractEmbed{F} end
abstract type AbstractBroadcastEmbed{F} <: AbstractEmbed{F} end

Base.eltype(::AbstractEmbed{F}) where F = F
_getF(::Type{<:AbstractEmbed{F}}) where F = F

struct CompositeEmbeddings{F, T <: NamedTuple, T2 <: NamedTuple} <: AbstractEmbed{F}
    embeddings::T
    aggregator::T2
    CompositeEmbeddings(es::NamedTuple, as::NamedTuple) = new{_getF(eltype(es)), typeof(es), typeof(as)}(es, as)
end

function CompositeEmbeddings(;es...)
    emb = map(x-> x isa Tuple ? x[1] : x, es.data)
    agg = map(x-> x isa Tuple ? x[2] : +, es.data)
    CompositeEmbeddings(emb, agg)
end
function Base.show(io::IO, e::CompositeEmbeddings)
    print(io, "CompositeEmbeddings")
    print(io, e.embeddings)
end

get_value(e::AbstractEmbed, name::Symbol, xs::NamedTuple) = e(xs[name])

_repeatdims(bs, vs) = map(x-> x[1] > length(vs) ? x[2] : 1, enumerate(bs))

aggregate(e::AbstractEmbed{T}, f::F, base::A, value::A) where {F, T, A <: AbstractArray{T}} = f(base, value)
aggregate(e::AbstractBroadcastEmbed{T}, f::F, base::A, value::A2) where {F, T, A <: AbstractArray{T}, A2 <: AbstractArray{T}} = f.(base, value)
aggregate(e::AbstractBroadcastEmbed{T}, f::typeof(vcat), base::A, value::A2) where {T, A <: AbstractArray{T}, A2 <: AbstractArray{T}} = cat(base, repeat(value, _repeatdims(size(base), size(value))...); dims=1)
aggregate(e::AbstractBroadcastEmbed{T}, f::typeof(hcat), base::A, value::A2) where {T, A <: AbstractArray{T}, A2 <: AbstractArray{T}} = cat(base, repeat(value, _repeatdims(size(base), size(value))...); dims=2)


(ce::CompositeEmbeddings)(;xs...) = ce(xs.data)
function (ce::CompositeEmbeddings{F})(xs::NamedTuple) where F
    names = keys(ce.embeddings)
    firstname = first(names)
    init = get_value(ce.embeddings[firstname], firstname, xs)

    for name ∈ Base.tail(names)
        emb = ce.embeddings[name]
        agg = ce.aggregator[name]
        value = get_value(emb, name, xs)
        init = aggregate(emb, agg, init, value)
    end

    init::TrackedArray{F}
end
