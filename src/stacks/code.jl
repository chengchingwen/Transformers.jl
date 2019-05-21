@inline issym(ex::Symbol) = true
@inline issym(ex) = false

@inline istuple(ex::Expr) = ex.head == :tuple
@inline istuple(ex) = false

@inline iscolon(ex::Expr) = ex.head == :call && first(ex.args) == :(:)
@inline iscolon(ex) = false

@inline iscollect(ex::Expr) = ex.head == Symbol("'")
@inline iscollect(ex) = false

@inline isleaf(ex::Expr) = istuple(ex) || iscolon(ex) || iscollect(ex)
@inline isleaf(ex::Symbol) = true
@inline isleaf(ex::Integer) = true
@inline isleaf(ex) = false

@inline getleft(ex::Expr) = ex.args[2]
@inline getright(ex::Expr) = ex.args[3]

leftmost(ex) = isleaf(ex) ? ex : leftmost(getleft(ex))
leftmostnode(ex) = isleaf(getleft(ex)) ? ex : leftmostnode(getleft(ex))

@inline isdup(ex::Integer) = true
@inline isdup(ex::Expr) = iscolon(ex) && getleft(ex) isa Integer
@inline isdup(ex) = false

@inline iscollectsym(ex) = iscollect(ex) && issym(first(ex.args))
@inline iscollecttup(ex) = iscollect(ex) && istuple(first(ex.args))

@inline istuplesym(ex) = istuple(ex) && all(issym, ex.args)
@inline istuplesymlike(ex) = istuple(ex) && all(issymlike, ex.args)

@inline issymlike(ex) = issym(ex) || iscollectsym(ex)
@inline istuplelike(ex) = istuplesymlike(ex) || iscollecttup(ex)

@inline hascollect(ex::Expr) = iscollect(ex) || any(iscollect, ex.args)
@inline hascollect(ex) = false

removecollect(ex) = ex
function removecollect(ex::Expr)
  if hascollect(ex)
    if iscollect(ex)
      return first(ex.args)
    else
      ret = copy(ex)
      for (i, e) ∈ enumerate(ex.args)
        if iscollect(e)
          ret.args[i] = first(e.args)
        end
      end
      return ret
    end
  else
    return ex
  end
end

collectcollect(ex) = :(())
function collectcollect(ex::Expr)
  if iscollect(ex)
    return first(ex.args)
  else
    ret = Any[]
    for (i, e) ∈ enumerate(ex.args)
      if iscollect(e)
        push!(ret, first(e.args))
      end
    end
    if length(ret) == 1
      return ret[1]
    else
      return Expr(:tuple, ret...)
    end
  end
end

const legalsym = (:(=>),:→)

@inline islegal(ex::Symbol) = true
@inline islegal(ex::Integer) = ex > 0
@inline islegal(ex) = false
function islegal(ex::Expr)
  global legalsym

  if istuple(ex)
    all(issymlike, ex.args)
  elseif iscolon(ex)
    length(ex.args) == 3 && isdup(ex) ?
      islegal(getleft(ex)) &&
      (issymlike(getright(ex)) ||
       (istuplelike(getright(ex)) && islegal(getright(ex)))) :
      (issymlike(getleft(ex)) ||
       (istuplelike(getleft(ex)) && islegal(getleft(ex)))) &&
      (issymlike(getright(ex)) ||
       (istuplelike(getright(ex)) && islegal(getright(ex))))
  elseif iscollect(ex)
    length(ex.args) == 1 && (issym(first(ex.args)) || istuplesym(first(ex.args)))
  else
    ex.head == :call &&
      first(ex.args) ∈ legalsym &&
      length(ex.args) == 3 &&
      islegal(getleft(ex)) &&
      islegal(getright(ex))
  end
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

@inline out(code::Code) = code.out
@inline out(c) = c

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

function duplicate(code::Code, n::Int)
  in = code.in
  out = code.out

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

@inline getint(ex::Int) = ex
@inline getint(ex::Expr) = getleft(ex) isa Int ? getleft(ex) : getright(ex)
@inline getsym(ex::Expr) = getleft(ex) isa Int ? getright(ex) : getleft(ex)

@inline getin(ex::Expr) = iscolon(ex) ? getleft(ex) : ex
@inline getin(x::Symbol) = x
@inline getout(ex::Expr) = iscolon(ex) ? getright(ex) : ex
@inline getout(x::Symbol) = x


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
