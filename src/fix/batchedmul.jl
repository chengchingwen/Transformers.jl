include("./batched_gemm.jl")

using Flux.Tracker
using Flux.Tracker: TrackedArray, track, data, @grad

const Tracked3D{T,A} = TrackedArray{T,3,A}

function batchedmul(a::Abstract3DTensor{T}, b::Abstract3DTensor{T};
                    transA::Bool = false, transB::Bool = false) where {T}
    (bs = size(a, 3)) == size(b, 3) || error("batch size mismatch")
    res = similar(a, size(a, transA ? 2 : 1), size(b, transB ? 1 : 2), bs)
    batched_mul!(res, a, b;transA=transA, transB=transB)
    return res
end

function batched_mul!(C::Abstract3DTensor{T}, A::Abstract3DTensor{T}, B::Abstract3DTensor{T};
                      transA::Bool = false, transB::Bool = false) where T
    At = transA ? 'T' : 'N'
    Bt = transB ? 'T' : 'N'
    batched_gemm!(At, Bt, one(T), A, B, zero(T), C)
    C
end

batchedmul(a::Tracked3D, b::Tracked3D; kw...) = track(batchedmul, a, b; kw...)
batchedmul(a::Abstract3DTensor, b::Tracked3D; kw...) = track(batchedmul, a, b; kw...)
batchedmul(a::Tracked3D, b::Abstract3DTensor; kw...) = track(batchedmul, a, b; kw...)

@grad function batchedmul(a::Abstract3DTensor, b::Abstract3DTensor; transA::Bool = false, transB::Bool = false)
    batchedmul(data(a), data(b); transA=transA, transB=transB),
    if transA
        if transB
            Δ -> (batchedmul(data(b), Δ; transA=true, transB=true), batchedmul(Δ, data(a); transA=true, transB=true))
        else
            Δ -> (batchedmul(data(b), Δ; transB=true), batchedmul(data(a), Δ))
        end
    else
        if transB
            Δ -> (batchedmul(Δ, data(b)), batchedmul(Δ, data(a); transA=true))
        else
            Δ -> (batchedmul(Δ, data(b); transB=true), batchedmul(data(a), Δ; transA=true))
        end
    end
end

