using JSON

json_load(f) = Dict{Symbol, Any}(Symbol(k)=>v for (k, v) in JSON.parsefile(f))

_ensure(f::Function, A, args...; kwargs...) = isnothing(A) ? f(args...; kwargs...) : A
