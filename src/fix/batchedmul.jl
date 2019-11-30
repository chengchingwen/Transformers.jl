include("./batched_gemm.jl")

using ZygoteRules: @adjoint


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

@adjoint function batchedmul(a::Abstract3DTensor, b::Abstract3DTensor; transA::Bool = false, transB::Bool = false)
    batchedmul(a, b; transA=transA, transB=transB),
    if transA
        if transB
            Δ -> (batchedmul(b, Δ; transA=true, transB=true), batchedmul(Δ, a; transA=true, transB=true))
        else
            Δ -> (batchedmul(b, Δ; transB=true), batchedmul(a, Δ))
        end
    else
        if transB
            Δ -> (batchedmul(Δ, b), batchedmul(Δ, a; transA=true))
        else
            Δ -> (batchedmul(Δ, b; transB=true), batchedmul(a, Δ; transA=true))
        end
    end
end

