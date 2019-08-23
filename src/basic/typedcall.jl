# using Tracker

# struct Sa{T, N,S, F, A <:AbstractArray{T, N}} <: AbstractArray{T, N}
#   cache::F
# end

# Sa(xs...) = Sa{xs...}()
# Sa(xs::Tuple) =Sa(xs...)
# Sa(a::AbstractArray{T,N}) where {T,N}= Sa(T,N, size(a), typeof(identity), typeof(a))

# info(s::Sa{T, N, S, F, A}) where {T, N, S, F, A} = (T, N, S, F, A)
# replace_info_f(i::Tuple, f) = (i[1], i[2], i[3], f, i[5])
# replace_info_s(i::Tuple, s) = (i[1], i[2], s, i[4], i[5])
# Base.similar(s::Sa) = Sa(info(s)...)

# #array_promotion(xs::Vararg{<:AbstractArray}...) = typeof(xs)
# Base.size(s::Sa) = info(s)[3]
# Base.getindex(s::Sa, ind...) = nothing
# Base.:(+)(a::Sa, b::Sa) = Sa(replace_info_f(info(a), +))

# delaycall(f, xs...; kw...) = () -> f(map(forward, xs)...; kw...)

# struct DelayArray{T, N, F} <: AbstractArray{T, N}
#   f::F
# end
# DelayArray(t, n, f) = DelayArray{t, n , typeof(f)}(f)
# Base.getindex(d::DelayArray) = d.f()

# struct TraceArray{T, N, S, A <: AbstractArray{T, N}, F} <: AbstractArray{T, N}
#   f::F
# end
# TraceArray{T,N,S}(a::AbstractArray) where {T,N,S} = (f=()->a; TraceArray{T, N, S, typeof(a), typeof(f)}(f))
# TraceArray{T,N,S}(a::F) where {T,N,S,F} = (f=()->a; TraceArray{T, N, S, typeof(a), typeof(f)}(f))
# TraceArray(a::AbstractArray{T, N}) where {T, N} = TraceArray{T, N, size(a)}(a)

# forward(t::TraceArray) = t.f()
# forward(t) = t

# Base.getindex(t::TraceArray) = t.data[]
# Base.size(t::TraceArray{T,N,S}) where {T, N, S} = S
# Base.:(+)(a::TraceArray{T, N, S}, b::TraceArray{T, N, S}) where {T,N,S} = TraceArray{T, N, S}(delaycall(+, a, b))
# Base.show(io::IO, t::TraceArray) = print(io, typeof(t))
# Base.print_array(io::IO, t::TraceArray) = io

