using Flux: Dropout, rand!, _dropout_kernel

_dropout_shape(s, dims...) = tuple((i ∈ dims ? 1 : si for (i, si) ∈ enumerate(s))...)

function (a::Dropout)(x, dims)
  a.active || return x
  y = similar(x, _dropout_shape(size(x), dims...)...)
  rand!(y)
  y .= _dropout_kernel.(y, a.p, 1 - a.p)
  return x .* y
end
