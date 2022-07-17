using JSON

json_load(f) = JSON.parsefile(f; dicttype = Dict{Symbol, Any})

_ensure(f::Function, A, args...; kwargs...) = isnothing(A) ? f(args...; kwargs...) : A
