@static if VERSION < v"1.8"
    macro etotal(ex)
        return :(Base.@pure $ex)
    end
else
    macro etotal(ex)
        return :(Base.@assume_effects :total $ex)
    end
end

@etotal function sym_in(x::Symbol, xs::Tuple{Vararg{Symbol}})
    @nospecialize xs
    for i = 1:length(xs)
        x == xs[i] && return i
    end
    return 0
end

@etotal function prefix_name(prefix::Symbol, names::Tuple{Vararg{Symbol}})
    @nospecialize names
    return map(Base.Fix1(Symbol, Symbol(prefix, :_)), names)
end

@etotal function replace_name(names::Tuple{Vararg{Symbol}}, a::Symbol, b::Symbol)
    @nospecialize names
    return map(name -> name == a ? b : name, names)
end

@etotal function replace_names(names::Tuple{Vararg{Symbol}}, as::NTuple{N, Symbol}, bs::NTuple{N, Symbol}) where N
    @nospecialize names as bs
    for i in Base.OneTo(N)
        names = replace_name(names, as[i], bs[i])
    end
    return names
end

@etotal function remove_name(names::Tuple{Vararg{Symbol}}, name::Symbol)
    @nospecialize names
    i = sym_in(name, names)
    return i == 0 ? names : (names[1:i-1]..., names[i+1:end]...)
end

function rename(nt::NamedTuple{names, types}, _a::Val{a}, _b::Val{b}) where {names, types, a, b}
    if iszero(sym_in(b, names))
        new_names = replace_name(names, a, b)
        return NamedTuple{new_names, types}(values(nt))
    else
        nt = Base.structdiff(nt, NamedTuple{(b,)})
        return rename(nt, _a, _b)
    end
end

function with_prefix(::Val{prefix}, nt::NamedTuple{names, types}) where {prefix, names, types}
    new_names = prefix_name(prefix, names)
    return NamedTuple{new_names, types}(values(nt))
end
with_prefix(prefix::Val) = Base.Fix1(with_prefix, prefix)


function _showm(T)
    return quote
        function Base.show(io::IO, x::$(esc(T)))
            print(io, nameof(typeof(x)))
            fields = ntuple(i->getfield(x, i), fieldcount(typeof(x)))
            for (i, field) in enumerate(fields)
                print(io, isone(i) ? "(" : ", ")
                show(io, field)
            end
            print(io, ')')
        end
    end
end
macro fluxshow(T, full=true)
    fullshow = full ? _showm(T) : nothing
    return quote
        function Base.show(io::IO, m::MIME"text/plain", x::$(esc(T)))
            if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
                Flux._big_show(io, x)
            elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
                Flux._layer_show(io, x)
            else
                show(io, x)
            end
        end
        $fullshow
    end
end
macro fluxlayershow(T, full=true)
    fullshow = full ? _showm(T) : nothing
    return quote
        function Base.show(io::IO, m::MIME"text/plain", x::$(esc(T)))
            if !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
                Flux._layer_show(io, x)
            else
                show(io, x)
            end
        end
        Flux._big_show(io::IO, layer::$(esc(T)), indent::Int, name = nothing) = Flux._layer_show(io, layer, indent, name)
        $fullshow
    end
end
