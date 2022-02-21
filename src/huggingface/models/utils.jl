using Statistics
import Flux

using ..Transformers.Basic: gather, OneHot, OneHotArray

import Flux.Losses

Base.getindex(a::AbstractVector, o::OneHot{K}) where K = a[Int(o)]
Base.getindex(a::AbstractArray, o::OneHot{K}, i...) where K = a[Int(o), i...]

function Base.getindex(xs::AbstractArray{T}, onehots::OneHotArray{K}) where {T, K}
  @assert size(xs, 1) == K
  ys = fill!(similar(xs, Base.tail(size(xs))), zero(T))

  Threads.@threads for i = 1:length(ys)
    @inbounds if (o = Int(onehots.data[i])) != 0
      ind = Tuple(CartesianIndices(onehots.data)[i])
      ys[ind...] = xs[o, ind...]
    end
  end
  return ys
end

using Flux: @adjoint

@adjoint function Base.getindex(xs::AbstractArray, onehots::OneHotArray{K}) where K
  xs[onehots], Δ -> begin
    dxs = fill!(similar(xs), zero(eltype(xs)))
    (∇getindex!(dxs, Δ, onehots), nothing)
  end
end

function ∇getindex!(xs, ys, onehots::OneHotArray{K}) where K
  Threads.@threads for i = 1:length(ys)
    @inbounds if (o = Int(onehots.data[i])) != 0
      ind = Tuple(CartesianIndices(onehots.data)[i])
      xs[o, ind...] = ys[ind...]
    end
  end
  return xs
end

import CUDA
import CUDA: CuVector, CuArray, CuDeviceArray, @cuda

function Base.getindex(xs::CuArray{T}, onehots::OneHotArray{K, N, N2, <:CuArray{OneHot{K}}}) where {T, K, N, N2}
  @assert size(xs, 1) == K
  ys = CUDA.zeros(T, size(onehots.data))

  function kernel(ys::CuDeviceArray{T}, xs::CuDeviceArray{T}, onehots)
    i = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x

    @inbounds if i <= length(onehots) && (o = Int(onehots[i])) != 0
      ind = Tuple(CartesianIndices(onehots)[i])
      ys[ind...] = xs[o, ind...]
    end
    return
  end

  function configurator(kernel)
    config = launch_configuration(kernel.fun)

    threads = Base.min(length(onehots.data), config.threads)
    blocks = cld(length(onehots.data), threads)

    return (threads=threads, blocks=blocks)
  end

  kernel = @cuda name="onehot_getindex" launch=false kernel(ys, xs, onehots.data)
  config = configurator(kernel)
  kernel(ys, xs, onehots.data; config...)
  return ys
end

function ∇getindex!(xs::CuArray{T}, ys::CuArray{T}, onehots::OneHotArray{K, N, N2, <:CuArray{OneHot{K}}}) where {T, K, N, N2}
  function kernel(xs::CuDeviceArray{T}, ys::CuDeviceArray{T}, onehots)
    i = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x

    @inbounds if i <= length(onehots) && (o = Int(onehots[i])) != 0
      ind = Tuple(CartesianIndices(onehots)[i])
      xs[o, ind...] = ys[ind...]
    end
    return
  end

  function configurator(kernel)
    config = launch_configuration(kernel.fun)

    threads = Base.min(length(onehots.data), config.threads)
    blocks = cld(length(onehots.data), threads)

    return (threads=threads, blocks=blocks)
  end

  kernel = @cuda name="onehot_getindex_grad" launch=false kernel(xs, ys, onehots.data)
  config = configurator(kernel)
  kernel(xs, ys, onehots.data; config...)
  return xs
end

struct MaskFillArray{T, N, S<:NTuple{N, Int},
                   L<:AbstractVector{T},
                   I<:NTuple{N, <:Integer}, P<:AbstractVector{I}} <: AbstractArray{T, N}
  size::S
  masked_val::T
  values::L
  positions::P
end

MaskFillArray(size, masked_val::T) where T = MaskFillArray(size, masked_val, Vector{T}(), Vector{NTuple{length(size), Int}}())

Base.size(ma::MaskFillArray) = ma.size

function Base.getindex(ma::MaskFillArray{T, N}, i::Vararg{Int, N}) where {T, N}
  @boundscheck checkbounds(ma, i...)
  idx = findfirst(isequal(i), ma.positions)
  if isnothing(idx)
    return ma.masked_val
  else
    return ma.values[idx]
  end
end

function Base.getindex(xs::CuArray{T}, indices::CuVector{<:Integer}) where T
  ys = CuArray{T}(undef, length(indices))

  function kernel(ys::CuDeviceArray{T}, xs, indices)
    i = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x

    @inbounds if i <= length(indices)
      idx = indices[i]
      ys[i] = xs[idx]
    end
    return
  end

  function configurator(kernel)
    config = launch_configuration(kernel.fun)

    threads = Base.min(length(indices), config.threads)
    blocks = cld(length(indices), threads)

    return (threads=threads, blocks=blocks)
  end

  kernel = @cuda name="linear_getindex" launch=false kernel(ys, xs, indices)
  config = configurator(kernel)
  kernel(ys, xs, indices; config...)
  return ys
end

function Base.getindex(ma::MaskFillArray{T, N}, indices::Vararg{Union{Colon, UnitRange, StepRange, Int}, N}) where {T, N}
  @boundscheck checkbounds(ma, indices...)

  indices_region = map(size(ma), indices) do si, ii
    if ii isa Int
      0
    elseif ii isa Colon
      si
    else
      length(ii)
    end
  end
  remaining_dim = map(!isequal(0), indices_region)
  new_size = filter(!iszero, indices_region)

  function transform_idx(old_idx)
    new_idx = map(old_idx, indices) do idx, ii
      if ii isa Int
        0
      elseif ii isa Colon
        idx
      else
        findfirst(isequal(idx), ii)
      end
    end
    filter(!iszero, new_idx)
  end

  _match(x::Tuple{Colon, Int}) = true
  _match(x::Tuple{Int, Int}) = x[1] == x[2]
  _match(x) = x[2] in x[1]
  match(idx) = all(_match, zip(indices, idx))

  if ma.positions isa CUDA.CuArray
    CUDA.@allowscalar idx = findall(match, ma.positions)
  else
    idx = findall(match, ma.positions)
  end

  new_values = ma.values[idx]
  new_positions = ma.positions[idx]
  if ma.positions isa CUDA.CuArray
    CUDA.@allowscalar new_positions = map(transform_idx, new_positions)
  else
    new_positions = map(transform_idx, new_positions)
  end

  MaskFillArray(new_size, ma.masked_val, new_values, new_positions)
end

function Base.setindex!(ma::MaskFillArray{T, N}, v, i::Vararg{Int, N}) where {T, N}
  @boundscheck checkbounds(ma, i...)
  idx = findfirst(isequal(i), ma.positions)
  if isnothing(idx)
    push!(ma.values, v)
    push!(ma.positions, i)
    return v
  else
    ma.values[idx] = v
  end
end

function MaskFillArray(oma::MaskFillArray{OneHot{K}}) where K
  new_size = (K, size(oma)...)
  new_values = map(x->true, oma.values)
  new_positions = map(zip(oma.values, oma.positions)) do (val, pos)
    (convert(Int, val), pos...)
  end
  MaskFillArray(new_size, false, new_values, new_positions)
end

Flux.@nograd MaskFillArray

import Adapt: adapt, adapt_structure
adapt_structure(T, ma::MaskFillArray) = MaskFillArray(ma.size, ma.masked_val, adapt(T, ma.values), adapt(T, ma.positions))

num_effective_entry(y::OneHotArray) = sum(map(!iszero, y.data))
Flux.@nograd num_effective_entry

function Losses.crossentropy(ŷ::AbstractArray{T, N}, y::OneHotArray{K, N}; agg=mean, ϵ=Flux.epseltype(ŷ)) where {T, K, N}
  ce = (.-log.(ŷ .+ ϵ))[y]
  if agg === sum || agg === mean
    ce = sum(ce)
    if agg === mean
      n = convert(T, num_effective_entry(y))
      return ce / n
    else
      return ce
    end
  else
    return agg(ce)
  end
end

function Losses.logitcrossentropy(ŷ::AbstractArray{T, N}, y::OneHotArray{K, N}; agg=mean) where {T, N, K}
  ce = .-logsoftmax(ŷ)[y]
  if agg === sum || agg === mean
    ce = sum(ce)
    if agg === mean
      n = convert(T, num_effective_entry(y))
      return ce / n
    else
      return ce
    end
  else
    return agg(ce)
  end
end

function Losses.crossentropy(ŷ, y::MaskFillArray{OneHot{K}}; agg=mean, ϵ=Flux.epseltype(ŷ)) where K
  real_ŷ = gather(ŷ, y.positions)
  real_y = OneHotArray(y.values)
  Losses.crossentropy(real_ŷ, real_y; agg=agg)
end

function Losses.logitcrossentropy(ŷ, y::MaskFillArray{OneHot{K}}; agg=mean) where K
  real_ŷ = gather(ŷ, y.positions)
  real_y = OneHotArray(y.values)
  Losses.logitcrossentropy(real_ŷ, real_y; agg=agg)
end

