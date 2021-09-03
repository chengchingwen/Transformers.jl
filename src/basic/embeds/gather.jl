"
    gather(w::AbstractMatrix{T}, xs::OneHotArray) where

getting vector at the given onehot encoding.
"
gather(w::AbstractMatrix{T}, xs::OneHotArray) where T = gather(w, onehot2indices(xs))
