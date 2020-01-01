using LinearAlgebra: tril!, triu!
function batched_tril!(A::CuArray{T, N}, d) where {T, N}
  if N < 2
    error("MethodError: no method matching tril!(::Array{Float64,1})")
  elseif N == 2
    return tril!(x, d)
  else
    s = size(A)
    m, n = s[1], s[2]
    l = m*n
    bs = Int(length(A) // l)
    function batch_tril_kernel!(_A, _d)
      li = (blockIdx().x - 1) * blockDim().x + threadIdx().x
      b = (blockIdx().y - 1) * blockDim().y + threadIdx().y
      @inbounds if 0 < li <= l && b <= bs
        id = Tuple(CartesianIndices(_A)[Base._to_linear_index(_A, mod1(li, m), fld1(li, m), b)])
        i, j = id
        if i < j - _d
          _A[id...] = 0
        end
      end
      return nothing
    end
    max_threads = 256
    thread_x = min(max_threads, l)
    thread_y = min(max_threads ÷ thread_x, bs)
    threads = (thread_x, thread_y)
    blocks = ceil.(Int, (l, bs) ./ threads)
    @cuda threads=threads blocks=blocks batch_tril_kernel!(A, d)
    return A
  end
end

function batched_triu!(A::CuArray{T, N}, d) where {T, N}
  if N < 2
    error("MethodError: no method matching triu!(::Array{Float64,1})")
  elseif N == 2
    return tril!(x, d)
  else
    s = size(A)
    m, n = s[1], s[2]
    l = m*n
    bs = Int(length(A) // l)
    function batch_triu_kernel!(_A, _d)
      li = (blockIdx().x - 1) * blockDim().x + threadIdx().x
      b = (blockIdx().y - 1) * blockDim().y + threadIdx().y
      @inbounds if 0 < li <= l && b <= bs
        id = Tuple(CartesianIndices(_A)[Base._to_linear_index(_A, mod1(li, m), fld1(li, m), b)])
        i, j = id
        if j < i + _d
          _A[id...] = 0
        end
      end
      return nothing
    end
    max_threads = 256
    thread_x = min(max_threads, l)
    thread_y = min(max_threads ÷ thread_x, bs)
    threads = (thread_x, thread_y)
    blocks = ceil.(Int, (l, bs) ./ threads)
    @cuda threads=threads blocks=blocks batch_triu_kernel!(A, d)
    return A
  end
end
