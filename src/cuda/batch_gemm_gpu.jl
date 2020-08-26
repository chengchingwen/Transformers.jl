#batched CuArray gemm by BatchedRoutines.jl
function batched_gemm!(transA::AbstractChar,
                       transB::AbstractChar,
                       alpha::Float32,
                       A::CUDA.CuArray{Float32, 3},
                       B::CUDA.CuArray{Float32, 3},
                       beta::Float32,
                       C::CUDA.CuArray{Float32, 3})
    CUDA.CUBLAS.gemm_strided_batched!(transA, transB, alpha, A, B, beta, C)
end

function batched_gemm!(transA::AbstractChar,
                       transB::AbstractChar,
                       alpha::Float64,
                       A::CUDA.CuArray{Float64, 3},
                       B::CUDA.CuArray{Float64, 3},
                       beta::Float64,
                       C::CUDA.CuArray{Float64, 3})
    CUDA.CUBLAS.gemm_strided_batched!(transA, transB, alpha, A, B, beta, C)
end
