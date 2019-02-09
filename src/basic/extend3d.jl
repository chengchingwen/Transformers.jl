import Flux: logsoftmax, Dense

"""

    @toNd f(x, y, z; a=a, b=b, c=c) n

macro for calling 2-d array function on N-d array by reshape input with reshape(x, size(x, 1), :)
and reshape back with reshape(out, :, input[n][2:end]...) where n is the n-th input(default=1).

"""
macro toNd(ex, outref::Int=1)
    fname = esc(ex.args[1])
    fkw = ex.args[2] isa Expr && ex.args[2].head == :parameters ? ex.args[2] : nothing
    fargs = esc.(fkw === nothing ? ex.args[2:end] : ex.args[3:end])
    fsize = map(x->Expr(:call, :size, x), fargs)
    rfargs = map((x, s) -> Expr(:call, :reshape, x, Expr(:ref, s, 1), :(:)), fargs, fsize)
    func = fkw === nothing ? Expr(:call, fname, rfargs...) : Expr(:call, fname, fkw, rfargs...)
    rsize = Expr(:ref, fsize[outref], Expr(:call , :(:), 2, :end))
    ret = Expr(:call, :reshape, func, :(:), Expr(:..., rsize))
    sT = gensym(:T)
    Expr(:(::), ret, Expr(:where,
                          Expr(:curly, :AbstractArray,
                               sT,
                               Expr(:call, :ndims, fargs[outref])),
                          sT)
         )
end


#extend Flux op for 3-dims input
function (a::LayerNorm)(x::ThreeDimArray{T}) where T
    s = size(x)
    reshape(a(reshape(x, s[1], :)), s)
end

function (d::Dense)(x::ThreeDimArray{T}) where T
    s = size(x)
    reshape(d(reshape(x, s[1], :)), size(d.W, 1), s[2], s[3])
end

#avoid ambiguity
(a::Dense{<:Any,W})(x::ThreeDimArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{ThreeDimArray}, x)

(a::Dense{<:Any,W})(x::ThreeDimArray{<:Real}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))


logsoftmax3d(x) = logsoftmax(x)
logsoftmax3d(x::ThreeDimArray) = @toNd logsoftmax(x)

