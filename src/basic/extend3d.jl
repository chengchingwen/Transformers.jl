using Base: tail

using Flux: applychain
import Flux: logsoftmax, Dense

"""

    @toNd f(x, y, z; a=a, b=b, c=c) n

macro for calling 2-d array function on N-d array by reshape input with reshape(x, size(x, 1), :)
and reshape back with reshape(out, :, input[n][2:end]...) where n is the n-th input(default=1).

"""
macro toNd(ex, outref::Int=1)
    fname = esc(ex.args[1])
    fkw = ex.args[2] isa Expr && ex.args[2].head == :parameters ? ex.args[2] : nothing
    _targs = Tuple(ex.args)
    fargs = esc.(fkw === nothing ? tail(_targs) : tail(tail(_targs)))
    fsize = map(x->Expr(:call, :size, x), fargs)
    rfargs = map((x, s) -> Expr(:call, :reshape, x, Expr(:ref, s, 1), :(:)), fargs, fsize)
    func = fkw === nothing ? Expr(:call, fname, rfargs...) : Expr(:call, fname, fkw, rfargs...)
    rsize = Expr(:call, :tail, fsize[outref])
    ret = Expr(:call, :reshape, func, :(:), Expr(:..., rsize))
    sT = gensym(:T)
    Expr(:(::), ret, Expr(:where,
                          Expr(:curly, :AbstractArray,
                               sT,
                               Expr(:call, :ndims, fargs[outref])),
                          sT)
         )
end

"""
    Sequence(layers)

just like Flux.Chain, but reshape input to 2d and reshape back when output.
"""
struct Sequence{T<:Tuple}
    models::T
    Sequence(xs...) = new{typeof(xs)}(xs)
end

Flux.children(s::Sequence) = s.models
Flux.mapchildren(f, s::Sequence) = Sequence(f.(s.models)...)

function (s::Sequence)(x)
    insize = size(x)
    y = applychain(s.models, reshape(x, insize[1], :))
    reshape(y, :, Base.tail(insize)...)
end
