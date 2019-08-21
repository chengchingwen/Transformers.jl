using Base: tail

using MacroTools: @forward, @capture
using Flux: applychain

"""
    @toNd f(x, y, z...; a=a, b=b, c=c...) n

macro for calling 2-d array function on N-d array by reshape input with reshape(x, size(x, 1), :)
and reshape back with reshape(out, :, input[n][2:end]...) where n is the n-th input(default=1).

"""
macro toNd(ex, outref::Int=1)
    if @capture ex f_(xs__; kw__)
        kwe = Expr(:parameters, kw...)
    else
        @capture ex f_(xs__)
    end
    rxs = map(xs) do x
        :(reshape($x, size($x, 1), :))
    end
    newcall = kw === nothing ? Expr(:call, f, rxs...) : Expr(:call, f, kwe, rxs...)
    :(reshape($newcall, :, Base.tail(size($(xs[outref])))...)) |> esc
end

"""
    Positionwise(layers)

just like `Flux.Chain`, but reshape input to 2d and reshape back when output. Work exactly the same as
`Flux.Chain` when input is 2d array.
"""
struct Positionwise{T<:Tuple}
    models::T
    Positionwise(xs...) = new{typeof(xs)}(xs)
end

@forward Positionwise.models Base.getindex, Base.length, Base.first, Base.last,
  Base.iterate, Base.lastindex

Flux.children(pw::Positionwise) = pw.models
Flux.mapchildren(f, pw::Positionwise) = Positionwise(f.(pw.models)...)

(pw::Positionwise)(x::A) where A <: AbstractMatrix = applychain(pw.models, x)
function (pw::Positionwise)(x)
    insize = size(x)
    y = applychain(pw.models, reshape(x, insize[1], :))
    reshape(y, :, Base.tail(insize)...)
end

function Base.show(io::IO, p::Positionwise)
  print(io, "Positionwise(")
  join(io, p.models, ", ")
  print(io, ")")
end
