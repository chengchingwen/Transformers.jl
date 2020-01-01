using LinearAlgebra: tril!, triu!

function batched_tril!(x::A, d) where {T, N, A <: AbstractArray{T, N}}
  if N < 2
    error("MethodError: no method matching tril!(::Array{Float64,1})")
  elseif N == 2
    return tril!(x, d)
  else
    s = size(x)
    m = (s[1], s[2])
    ms = s[1] * s[2]
    len = Int(length(x) // ms)
    Wt = Core.apply_type(A.name.wrapper, T, 2)
    Threads.@threads for i = 1:len
      tril!(Base.unsafe_wrap(Wt, Base.pointer(x, (ms * (i - 1) + 1)), m), d)
    end
    return x
  end
end

function batched_triu!(x::A, d) where {T, N, A <: AbstractArray{T, N}}
  if N < 2
    error("MethodError: no method matching triu!(::Array{Float64,1})")
  elseif N == 2
    return triu!(x, d)
  else
    s = size(x)
    m = (s[1], s[2])
    ms = s[1] * s[2]
    len = Int(length(x) // ms)
    Wt = Core.apply_type(A.name.wrapper, T, 2)
    Threads.@threads for i = 1:len
      triu!(Base.unsafe_wrap(Wt, Base.pointer(x, (ms * (i - 1) + 1)), m), d)
    end
    return x
  end
end
