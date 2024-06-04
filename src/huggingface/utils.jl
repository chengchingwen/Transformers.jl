using Mmap
using JSON3
using HuggingFaceApi: list_model_files

function mmap_open(file, fsz = filesize(file))
    return open(f->Mmap.mmap(f, Vector{UInt8}, (Int(fsz), )), file, "r")
end

json_load(f) = JSON3.read(mmap_open(f))

_ensure(f::Function, A, args...; kwargs...) = isnothing(A) ? f(args...; kwargs...) : A

ensure_possible_files(possible_files, model_name; revision = nothing, auth_token = HuggingFaceApi.get_token(), kw...) =
    _ensure(list_model_files, possible_files, model_name; revision, token = auth_token)

ensure_config(config, model_name; kw...) = _ensure(load_config, config, model_name; kw...)

load_error_msg(msg) = "$msg\nFile an issue with the model name you want to load."
load_error(msg) = error(load_error_msg(msg))

struct AliasNamedTuple{
    FN <: NamedTuple{NAME, T} where {NAME, T <: Tuple{Int, Vararg{Int}}},
    FT <: Tuple
}
    namemap::FN
    element::FT
end

Base.length(c::AliasNamedTuple) = length(getfield(c, :element))
Base.getindex(c::AliasNamedTuple, i::Integer) = getfield(c, :element)[i]
Base.getindex(c::AliasNamedTuple, k::Symbol) = c[getfield(c, :namemap)[k]]
Base.firstindex(c::AliasNamedTuple) = firstindex(getfield(c, :element))
Base.lastindex(c::AliasNamedTuple) = lastindex(getfield(c, :element))
Base.haskey(c::AliasNamedTuple, k) = haskey(getfield(c, :namemap), k)
Base.keys(c::AliasNamedTuple) = ntuple(i->keys(getfield(c, :namemap))[i], Val(length(c)))
Base.values(c::AliasNamedTuple) = getfield(c, :element)
Base.get(c::AliasNamedTuple, key, default) = haskey(c, key) ? c[key] : default
Base.propertynames(c::AliasNamedTuple) = keys(getfield(c, :namemap))
Base.getproperty(c::AliasNamedTuple, s::Symbol) = c[s]
Base.iterate(c::AliasNamedTuple, state...) = iterate(getfield(c, :element), state...)
Base.pairs(c::AliasNamedTuple) = Base.Pairs(c, keys(c))
Base.keytype(c::AliasNamedTuple) = Symbol
Base.valtype(c::AliasNamedTuple) = eltype(getfield(c, :element))
function Base.show(_io::IO, c::AliasNamedTuple)
    io = IOContext(_io, :compact => true, :limit => true)
    print(io, '(')
    n = length(c)
    for i = 1:n
        names = findall(==(i), getfield(c, :namemap))
        if length(names) == 1
            print(io, names[])
        else
            print(io, '[')
            join(io, names, ", ")
            print(io, ']')
        end
        print(io, " = ")
        show(io, c[i])
        i == n || print(io, ", ")
    end
    print(io, ')')
end

aliasof(namemap::NamedTuple{NAME, T}, k::Symbol) where {NAME, T <: Tuple{Int, Vararg{Int}}} = findfirst(==(get(namemap, k, 0)), namemap)
aliasof(c::AliasNamedTuple, k::Symbol) = aliasof(getfield(c, :namemap), k)

function aliases(namemap::NamedTuple{NAME, T}) where {NAME, T <: Tuple{Int, Vararg{Int}}}
    n = maximum(namemap)
    return keys(namemap)[n+1:end]
end
function aliases(c::AliasNamedTuple)
    namemap = getfield(c, :namemap)
    return keys(namemap)[length(c)+1:end]
end

aliasgroup(namemap::NamedTuple{NAME, T}, k::Symbol) where {NAME, T <: Tuple{Int, Vararg{Int}}} = findall(==(get(namemap, k, 0)), namemap)
aliasgroup(c::AliasNamedTuple, k::Symbol) = aliasgroup(getfield(c, :namemap), k)

function alias_tuplem(ex)
    Meta.isexpr(ex, :tuple) || error("syntax: expression is not a named tuple")
    namemap = Expr(:tuple)
    element = Expr(:tuple)
    idx = 0
    aliasnames = []
    for arg in ex.args
        Meta.isexpr(arg, :(=), 2) || error("syntax: invalid named tuple element: $arg")
        name, value = arg.args
        push!(element.args, esc(value))
        idx += 1
        if name isa Symbol
            push!(namemap.args, Expr(:(=), name, idx))
        elseif Meta.isexpr(name, :vect)
            length(name.args) > 0 || error("syntax: invalid named tuple element: name alias is empty")
            all(Base.Fix2(isa, Symbol), name.args) ||
                error("syntax: invalid named tuple element: aliases are not Symbols -> $(name.args)")
            mainname = popfirst!(name.args)
            restnames = name.args
            push!(namemap.args, Expr(:(=), mainname, idx))
            !isempty(restnames) && push!(aliasnames, (idx, restnames))
        else
            error("syntax: invalid named tuple element: $arg")
        end
    end
    for (idx, aliases) in aliasnames, alias in aliases
        push!(namemap.args, Expr(:(=), alias, idx))
    end
    return Expr(:call, :AliasNamedTuple, namemap, element)
end

macro alias(ex)
    return alias_tuplem(ex)
end
