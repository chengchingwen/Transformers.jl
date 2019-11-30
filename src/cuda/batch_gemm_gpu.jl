#batched CuArray gemm by BatchedRoutines.jl
function batched_gemm!(transA::AbstractChar,
                       transB::AbstractChar,
                       alpha::Float32,
                       A::CuArrays.CuArray{Float32, 3},
                       B::CuArrays.CuArray{Float32, 3},
                       beta::Float32,
                       C::CuArrays.CuArray{Float32, 3})
    CuArrays.CUBLAS.gemm_strided_batched!(transA, transB, alpha, A, B, beta, C)
end

function batched_gemm!(transA::AbstractChar,
                       transB::AbstractChar,
                       alpha::Float64,
                       A::CuArrays.CuArray{Float64, 3},
                       B::CuArrays.CuArray{Float64, 3},
                       beta::Float64,
                       C::CuArrays.CuArray{Float64, 3})
    CuArrays.CUBLAS.gemm_strided_batched!(transA, transB, alpha, A, B, beta, C)
end
