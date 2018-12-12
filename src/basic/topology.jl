
function _regex_escape(s::AbstractString)
    res = replace(s, r"([()[\]{}?*+\-|^\$\\.&~#\s=!<>|:])" => s"\\\1")
    Regex(replace(res, "\0" => "\\0"))
end


is_sublayer(x, ex) = startswith(String(ex), string(x, "_"))

const lr = :←
const rr = :→


lpass(name, arg::Symbol, m, i::Int) = Expr(:(=), name, Expr(:call, :($m[$i]), arg))
lpass(name, args::Expr, m, i::Int) = Expr(:(=), name, Expr(:call, :($m[$i]), args.args...))

macro topology(m, pattern)
    @show m, pattern
    # for i in ins.args
    #     is_sublayer(m, i) && @show i
    # end

    code = to_code(pattern)

    if isa(code.in, Symbol)
        fname = Expr(:call, m, code.in)
    else
        fname = Expr(:call, m, code.in.args...)
    end

    fbody = Any[:block]
    for (i, l) ∈ enumerate(code.lines)
        push!(fbody, lpass(l..., m, i))
    end

    push!(fbody, code.out)
    Expr(:function, fname, Expr(fbody...))
end


const legalsym = (:→,)# :←)
const fcache = Dict{String, Function}()

islegaltopo(ex) = false
islegaltopo(ex::Int) = true
islegaltopo(ex::Symbol) = true
islegaltopo(ex::Expr) = (@show ex; ex.head == :call ?
                         ex.args[1] ∈ legalsym &&
                         length(ex.args) == 3 &&
                         islegaltopo(ex.args[2]) &&
                         islegaltopo(ex.args[3]) : ex.head == :tuple &&
                         length(ex.args) == 2 &&
                         (isa(ex.args[1], Symbol) || isa(ex.args[1], Int)) &&
                         (isa(ex.args[2], Symbol) || isa(ex.args[2], Int))
)
struct NetTopo
    fs::String
    NetTopo(s::String) = (tofunc(s); new(s))
end

(nt::NetTopo)(xs...) = (global cache; cache[nt.fs](xs...))


macro nettopo_str(str)
    NetTopo(str)
end



function tofunc(sf::String)
    # global cache
    # haskey(cache, sf) && (println("cache found"); return cache[sf])

    ex = Meta.parse(sf)
    code = to_code(ex)
    

end

#=
x => b => c  ==> b = m[1](x) ; c = m[2](b)
x => 3 ==> x => a => a => a ==> x = m[1](a); a = m[1](a); a = m[1](a)
(x, m) => a => b => c ==> a = m[1](x , m); b = m[2](b); c = m[3](b)
((x, m) => x) => 3 ==> (x = m[1](x, m)); (x = m[2](x, m)); (x = m[3](x, m))


=#

struct Code
    in
    out
    lines::Vector{Tuple}
end

Code(in, out) = Code(in, out, [(out, in)])

function Base.show(io::IO, code::Code)
    println(io, "Code: input: $(code.in), output: $(code.out)")
    println(io, "body: ")
    for l in code.lines
        println("\t$(l[1]) = $(l[2])")
    end
    io
end

out(code::Code) = code.out
out(c) = c

function add(code1::Code, code2::Code)
    in = code1.in
    out = code2.out
    lines = cat(code1.lines, code2.lines; dims=1)
    Code(in, out, lines)
end

function add(code::Code, newline)
    in = code.in
    out = newline
    lines = [code.lines..., (out, code.out)]
    Code(in, out, lines)
end

function add(left, right)
    in = left
    out = right
    Code(in, out)
end




isleaf(ex::Expr) = ex.head == :tuple || ex.head == Symbol("'")
isleaf(ex::Symbol) = true
isleaf(ex::Int) = true

leftmost(ex) = isleaf(ex) ? ex : leftmost(ex.args[2])

leftmostnode(ex) = isleaf(ex.args[2]) ? ex : leftmostnode(ex.args[2])

function duplicate(code::Code, n::Int)
    in = code.in
    out = code.out
    n <= 0 && error("n should > 0")
    typeof(in) != typeof(out) && error("input number inconsistent")

    tp_cls = [(first(code.lines)[1], out), code.lines[2:end]...]
    tp_cls = [tp_cls for i = 1:(n-1)]

    Code(in, out, cat(code.lines, tp_cls...; dims=1))
end

function duplicate(c, n::Int)
    in = c
    out = c
    lines = [(c, c) for i = 1:n]
    Code(in, out, lines)
end


function to_code(node)
    if !isleaf(node.args[2])
        pre_code = to_code(node.args[2])
    else
        pre_code = node.args[2]
    end

    rL = leftmost(node.args[3]) #Symbol(or tuple) or Int
    if isa(rL, Int)
        if isa(node.args[3], Int)
            node.args[3] = out(pre_code)
        else
            leftmostnode(node.args[3]).args[2] = out(pre_code)
        end
        if rL == 1
            code = pre_code
        else
            code = duplicate(pre_code, rL)
        end
    else
        code = add(pre_code, rL)
    end

    if !isleaf(node.args[3])
        next_code = to_code(node.args[3])
        code = add(code, next_code)
    end

    code
end
