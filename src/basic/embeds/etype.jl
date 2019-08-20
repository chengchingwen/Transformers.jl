abstract type AbstractEmbed{F} end
abstract type AbstractBroadcastEmbed{F} <: AbstractEmbed{F} end

Base.eltype(::AbstractEmbed{F}) where F = F
_getF(::Type{<:AbstractEmbed{F}}) where F = F

struct CompositeEmbedding{F, T <: NamedTuple, T2 <: NamedTuple, P} <: AbstractEmbed{F}
    embeddings::T
    aggregator::T2
    postprocessor::P
    CompositeEmbedding(es::NamedTuple, as::NamedTuple, post) = new{_getF(eltype(es)), typeof(es), typeof(as), typeof(post)}(es, as, post)
end

"""
    CompositeEmbedding(;postprocessor=identity, es...)

composite several embedding into one embedding according the aggregate methods and apply `postprocessor` on it.
"""
function CompositeEmbedding(;postprocessor=identity, es...)
  eas = es.data
  emb = map(x-> x isa Tuple ? x[1] : x, eas)
  agg = map(x-> x isa Tuple ? x[2] : +, eas)
  CompositeEmbedding(emb, agg, postprocessor)
end

Flux.children(ce::CompositeEmbedding) = tuple(values(ce.embeddings)..., ce.postprocessor)
Flux.mapchildren(f, ce::CompositeEmbedding) = CompositeEmbedding(map(e->f(e), ce.embeddings), ce.aggregator, f(ce.postprocessor))

function Base.show(io::IO, e::CompositeEmbedding)
    names = keys(e.embeddings)
    firstname = first(names)

    print(io, "CompositeEmbedding(")
    print(io, firstname)
    print(io, " = ")
    print(io, e.embeddings[firstname])

    for name ∈ Base.tail(names)
        print(io, ", ")
        print(io, name)
        print(io, " = ")
        print(io, e.embeddings[name])
    end

    if e.postprocessor !== identity
        print(io, ", postprocessor = ")
        print(io, e.postprocessor)
    end

    print(io, ")")
end

get_value(e::AbstractEmbed, name::Symbol, xs::NamedTuple) = e(xs[name])
_repeatdims(bs, vs) = ntuple(i -> i > length(vs) ? bs[i] : 1, length(vs))

@inline aggregate(e::E, f::F, base::A, value::A2) where {F, T, A <: AbstractArray{T}, A2 <: AbstractArray{T}, E <: AbstractEmbed{T}} = f(base, value)
@inline aggregate(e::Eb, f::F, base::A, value::A2) where {F, T, A <: AbstractArray{T}, A2 <: AbstractArray{T}, Eb <: AbstractBroadcastEmbed{T}} = f.(base, value)
@inline aggregate(e::Eb, f::typeof(vcat), base::A, value::A2) where {T, A <: AbstractArray{T}, A2 <: AbstractArray{T}, Eb <: AbstractBroadcastEmbed{T}} = vcat(base, repeat(value, _repeatdims(size(base), size(value))...))
@inline aggregate(e::Eb, f::typeof(hcat), base::A, value::A2) where {T, A <: AbstractArray{T}, A2 <: AbstractArray{T}, Eb <: AbstractBroadcastEmbed{T}} = hcat(base, repeat(value, _repeatdims(size(base), size(value))...))

(ce::CompositeEmbedding)(;xs...) = ce(xs.data)
function (ce::CompositeEmbedding{F, T1, T2, P})(xs::NamedTuple) where {F, T1, T2, P}
    names = keys(ce.embeddings)
    firstname = first(names)
    init = get_value(ce.embeddings[firstname], firstname, xs)

    for name ∈ Base.tail(names)
        emb = ce.embeddings[name]
        agg = ce.aggregator[name]
        value = get_value(emb, name, xs)
        init = aggregate(emb, agg, init, value)
    end

    if P === typeof(identity)
        init
    else
        ce.postprocessor(init)
    end
end
