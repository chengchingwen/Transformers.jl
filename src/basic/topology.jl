#=
examples
x => b => c  ==> b = m[1](x) ; c = m[2](b)
x => 3 ==> x => a => a => a ==> x = m[1](a); a = m[1](a); a = m[1](a)
(x, m) => a => b => c ==> a = m[1](x , m); b = m[2](b); c = m[3](b)
((x, m) => x) => 3 ==> (x = m[1](x, m)); (x = m[2](x, m)); (x = m[3](x, m))
(((x, m) => x:(x, m)) => 3) ==> (x = m[1](x,m)); (x = m[2](x,m)) ;(x = m[3](x,m))
=#

struct NNTopo{F}
    fs::String
    f::F
end

NNTopo(ex::Expr) = NNTopo(string(ex), tofunc(ex))
NNTopo(s::String) = NNTopo(s, tofunc(s))
(nt::NNTopo)(xs...) = nt.f(xs...)

macro nntopo_str(str)
    NNTopo(str)
end

macro nntopo(expr)
    NNTopo(string(expr), tofunc(expr))
end

tofunc(sf::String) = tofunc(Meta.parse(sf))
function tofunc(pattern)
    !islegal(pattern) && error("topo pattern illegal")

    m = gensym(:model)
    fname = gensym(:topo_func)

    code = to_code(pattern)

    if istuple(code.in)
        fname = Expr(:call, fname, m, code.in.args...)
    else
        fname = Expr(:call, fname, m, code.in)
    end

    fbody = Any[:block]
    for (i, l) ∈ enumerate(code.lines)
        push!(fbody, genline(l..., m, i))
    end

    push!(fbody, code.out)
    func = Expr(:function, fname, Expr(fbody...))

    eval(func)
end

function Base.show(io::IO, nt::NNTopo)
    println(io, "NNTopo{\"$(nt.fs)\"}")
    print_topo(io, nt)
    io
end

print_topo(nt::NNTopo; models=nothing) = print_topo(stdout, nt; models=models)
function print_topo(io::IO, nt::NNTopo; models=nothing)
    code = to_code(Meta.parse(nt.fs))
    farg = istuple(code.in) ? join(code.in.args, ", ") : string(code.in)
    println(io, "topo_func(model, $farg)")
    for (i, l) ∈ enumerate(code.lines)
        name = string(l[1])
        args = istuple(l[2]) ? string(l[2]) : "($(l[2]))"
        model = models === nothing ? "model[$i]" : string(models[i])
        println(io, "\t$name = $model$args")
    end
    println(io, "\t$(code.out)")
    println("end")
end


const legalsym = (:(=>),:→,:(:))

islegal(ex) = false
islegal(ex::Symbol) = true
islegal(ex::Int) = true
function islegal(ex::Expr)
    if istuple(ex)
        all(x->x isa Symbol, ex.args)
    elseif iscolon(ex)
        length(ex.args) == 3 && isdup(ex) ?
            getsym(ex) isa Symbol || istuple(getsym(ex)) :
            (ex.args[2] isa Symbol || istuple(ex.args[2])) &&
            (ex.args[3] isa Symbol || istuple(ex.args[3]))
    else
        ex.head == :call && ex.args[1] ∈ legalsym &&
            length(ex.args) == 3 && islegal(ex.args[2]) && islegal(ex.args[3])
    end
end



genline(name, arg::Symbol, m, i::Int) = Expr(:(=), name, Expr(:call, :($m[$i]), arg))
genline(name, args::Expr, m, i::Int) = Expr(:(=), name, Expr(:call, :($m[$i]), args.args...))

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

isdup(ex) = false
isdup(ex::Int) = true
isdup(ex::Expr) = iscolon(ex) && any(x-> x isa Int, ex.args) &&
    any(x -> istuple(x) || x isa Symbol, ex.args)

getint(ex::Int) = ex
getint(ex::Expr) = getleft(ex) isa Int ? getleft(ex) : getright(ex)
getsym(ex::Expr) = getleft(ex) isa Int ? getright(ex) : getleft(ex)

getleft(ex::Expr) = ex.args[2]
getright(ex::Expr) = ex.args[3]

function _to_code(node)
    if !isleaf(getleft(node))
        pre_code = _to_code(getleft(node))
    else
        pre_code = getleft(node)
    end

    rL = leftmost(getright(node)) #Symbol(or tuple) or Int
    if isdup(rL)#rL isa Int
        rightnode = getright(node)
        if isdup(rightnode)#getright(node) isa Int
            node.args[3] = rL isa Int ? out(pre_code) : getsym(rL)
        else
            leftmostnode(rightnode).args[2] = rL isa Int ? out(pre_code) : getsym(rL)
        end
        if getint(rL) == 1 #don't need duplicate
            code = pre_code
        else
            code = duplicate(pre_code, getint(rL))
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


# is_sublayer(x, ex) = startswith(String(ex), string(x, "_"))
# macro topology(pattern)
#     @show pattern
#     # for i in ins.args
#     #     is_sublayer(m, i) && @show i
#     # end
#     !islegal(pattern) && error("topo pattern illegal")

#     m = gensym(:model)
#     fname = gensym(:topo_func)

#     code = to_code(pattern)

#     if isa(code.in, Symbol)
#         fname = Expr(:call, fname, m, code.in)
#     else
#         fname = Expr(:call, fname, m, code.in.args...)
#     end

#     fbody = Any[:block]
#     for (i, l) ∈ enumerate(code.lines)
#         push!(fbody, genline(l..., m, i))
#     end

#     push!(fbody, code.out)
#     Expr(:function, fname, Expr(fbody...))
# end


