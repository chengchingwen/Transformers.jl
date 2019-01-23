using Flux.Tracker
using Flux.Tracker: TrackedArray, track, data, @grad

const Tracked2D{T,A} = TrackedArray{T,2,A}

#import LinearAlgebra: BLAS
import LinearAlgebra
import CuArrays

for elty in (:Float64, :Float32)
    @eval begin
        gemm!(tA::AbstractChar, tB::AbstractChar,
              alpha::($elty),
              A::AbstractArray{$elty, 2},
              B::AbstractArray{$elty, 2},
              beta::($elty),
              C::AbstractArray{$elty, 2}) =
                  LinearAlgebra.BLAS.gemm!(tA, tB, alpha, A, B, beta, C)

        gemm!(tA::AbstractChar, tB::AbstractChar,
              alpha::($elty),
              A::CuArrays.CuArray{$elty, 2},
              B::CuArrays.CuArray{$elty, 2},
              beta::($elty),
              C::CuArrays.CuArray{$elty, 2}) =
                  CuArrays.CUBLAS.gemm!(tA, tB, alpha, A, B, beta, C)
    end
end





function matmul(a::TwoDimArray{T}, b::TwoDimArray{T};
                    transA::Bool = false, transB::Bool = false) where {T}
    res = similar(a, size(a, transA ? 2 : 1), size(b, transB ? 1 : 2))
    matmul!(res, a, b;transA=transA, transB=transB)
    return res
end

function matmul!(C::TwoDimArray{T}, A::TwoDimArray{T}, B::TwoDimArray{T};
                      transA::Bool = false, transB::Bool = false) where T
    At = transA ? 'T' : 'N'
    Bt = transB ? 'T' : 'N'
    gemm!(At, Bt, one(T), A, B, zero(T), C)
    C
end

matmul(a::Tracked2D, b::Tracked2D; kw...) = track(matmul, a, b; kw...)
matmul(a::TwoDimArray, b::Tracked2D; kw...) = track(matmul, a, b; kw...)
matmul(a::Tracked2D, b::TwoDimArray; kw...) = track(matmul, a, b; kw...)

@grad function matmul(a::TwoDimArray, b::TwoDimArray; transA::Bool = false, transB::Bool = false)
    matmul(data(a), data(b); transA=transA, transB=transB),
    if transA
        if transB
            Δ -> (matmul(data(b), Δ; transA=true, transB=true), matmul(Δ, data(a); transA=true, transB=true))
        else
            Δ -> (matmul(data(b), Δ; transB=true), matmul(data(a), Δ))
        end
    else
        if transB
            Δ -> (matmul(Δ, data(b)), matmul(Δ, data(a); transA=true))
        else
            Δ -> (matmul(Δ, data(b); transB=true), matmul(data(a), Δ; transA=true))
        end
    end
end

