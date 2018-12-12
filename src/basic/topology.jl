#=
x => b => c  ==> b = m[1](x) ; c = m[2](b)
x => 3 ==> x => a => a => a ==> x = m[1](a); a = m[1](a); a = m[1](a)
(x, m) => a => b => c ==> a = m[1](x , m); b = m[2](b); c = m[3](b)
((x, m) => x) => 3 ==> (x = m[1](x, m)); (x = m[2](x, m)); (x = m[3](x, m))


=#


is_sublayer(x, ex) = startswith(String(ex), string(x, "_"))


genline(name, arg::Symbol, m, i::Int) = Expr(:(=), name, Expr(:call, :($m[$i]), arg))
genline(name, args::Expr, m, i::Int) = Expr(:(=), name, Expr(:call, :($m[$i]), args.args...))

macro topology(pattern)
    @show pattern
    # for i in ins.args
    #     is_sublayer(m, i) && @show i
    # end
    !islegal(pattern) && error("topo pattern illegal")

    m = gensym(:model)
    fname = gensym(:topo_func)

    code = to_code(pattern)

    if isa(code.in, Symbol)
        fname = Expr(:call, fname, m, code.in)
    else
        fname = Expr(:call, fname, m, code.in.args...)
    end

    fbody = Any[:block]
    for (i, l) ∈ enumerate(code.lines)
        push!(fbody, genline(l..., m, i))
    end

    push!(fbody, code.out)
    Expr(:function, fname, Expr(fbody...))
end


const legalsym = (:(=>),:→,:(:))

islegal(ex) = false
islegal(ex::Symbol) = true
islegal(ex::Int) = true
islegal(ex::Expr) = (@show ex; istuple(ex) ?
    all(x->x isa Symbol, ex.args) :
    iscolon(ex) ?
    length(ex.args) == 3 &&
    (ex.args[2] isa Symbol || istuple(ex.args[2])) &&
    (ex.args[3] isa Symbol || istuple(ex.args[3])) :
    ex.head == :call && ex.args[1] ∈ legalsym &&
    length(ex.args) == 3 && islegal(ex.args[2]) && islegal(ex.args[3])
)

struct NNTopo{F}
    fs::String
    f::F
end

NNTopo(s::String) = NNTopo(s, tofunc(s))
(nt::NNTopo)(xs...) = nt.f(xs...)


macro nntopo_str(str)
    NNTopo(str)
end



function tofunc(sf::String)
    pattern = Meta.parse(sf)
    !islegal(pattern) && error("topo pattern illegal")

    m = gensym(:model)
    fname = gensym(:topo_func)

    code = to_code(pattern)

    if isa(code.in, Symbol)
        fname = Expr(:call, fname, m, code.in)
    else
        fname = Expr(:call, fname, m, code.in.args...)
    end

    fbody = Any[:block]
    for (i, l) ∈ enumerate(code.lines)
        push!(fbody, genline(l..., m, i))
    end

    push!(fbody, code.out)
    @show func = Expr(:function, fname, Expr(fbody...))

    eval(func)
end


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

iscolon(ex::Expr) = ex.head == :call && ex.args[1] == :(:)
iscolon(x) = false
istuple(ex::Expr) = ex.head == :tuple
istuple(ex) = false

isleaf(ex::Expr) = istuple(ex) || iscolon(ex)
isleaf(ex::Symbol) = true
isleaf(ex::Int) = true

leftmost(ex) = isleaf(ex) ? ex : leftmost(ex.args[2])

leftmostnode(ex) = isleaf(ex.args[2]) ? ex : leftmostnode(ex.args[2])

function duplicate(code::Code, n::Int)
    in = code.in
    out = code.out
    n <= 0 && error("n should > 0")

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

getleft(ex::Expr) = ex.args[2]
getright(ex::Expr) = ex.args[3]

function _to_code(node)
    @show node
    if !isleaf(getleft(node))
        pre_code = _to_code(getleft(node))
    else
        pre_code = getleft(node)
    end

    rL = leftmost(getright(node)) #Symbol(or tuple) or Int
    if rL isa Int
        if getright(node) isa Int
            node.args[3] = out(pre_code)
        else
            leftmostnode(getright(node)).args[2] = out(pre_code)
        end
        if rL == 1 #don't need duplicate
            code = pre_code
        else
            code = duplicate(pre_code, rL)
        end
    else
        code = add(pre_code, rL)
    end

    if !isleaf(getright(node))
        next_code = _to_code(getright(node))
        code = add(code, next_code)
    end

    code
end

getin(ex::Expr) = iscolon(ex) ? getleft(ex) : ex
getin(x::Symbol) = x
getout(ex::Expr) = iscolon(ex) ? getright(ex) : ex
getout(x::Symbol) = x

function _postcode(code::Code)
    in = getin(code.in)
    out = getin(code.out)
    lines = map(code.lines) do (vab, farg)
        (getin(vab), getout(farg))
    end
    Code(in, out, lines)
end

function to_code(ex::Expr)
    code = _to_code(ex)
    _postcode(code)
end
