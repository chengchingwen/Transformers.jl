using Flux
using Flux: @treelike
using Flux.Tracker
using LinearAlgebra: LowerTriangular

struct MultiheadAttention
    head::Int
    future::Bool
    iqproj::Dense
    ikproj::Dense
    ivproj::Dense
    oproj::Dense
end

@treelike MultiheadAttention

MultiheadAttention(head::Int,
                   is::Int,
                   hs::Int,
                   os::Int; future::Bool=true) = MultiheadAttention(head,
                                                                    future,
                                                                    Dense(is, hs*head),
                                                                    Dense(is, hs*head),
                                                                    Dense(is, hs*head),
                                                                    Dense(hs*head, os))


function Base.show(io::IO, mh::MultiheadAttention)
    hs = div(size(mh.iqproj.W)[1], mh.head)
    is = size(mh.iqproj.W)[end]
    os = size(mh.oproj.W)[1]

    print(io, "MultiheadAttention(")
    print(io, "head=$(mh.head), ")
    print(io, "head_size=$(hs), ")
    print(io, "$(is)=>$(os))")
end

function (mh::MultiheadAttention)(query::AbstractArray{T, 3},
                                  key::AbstractArray{T, 3},
                                  value::AbstractArray{T, 3};
                                  mask=nothing) where T
    qs = size(query)
    ks = size(key)
    vs = size(value)

    #size(ipq) == (h, q_seq_len * batch)
    ipq = mh.iqproj(reshape(query, qs[1], :))
    ipk = mh.ikproj(reshape(key, ks[1], :))
    ipv = mh.ivproj(reshape(value, vs[1], :))

    h = size(ipq, 1)
    hs = div(h, mh.head)

    #size(ipq) == (hs, q_seq_len, head, batch)
    ipq = permutedims(reshape(ipq, :, mh.head, qs[2], qs[3]), [1, 3, 2, 4])
    ipk = permutedims(reshape(ipk, :, mh.head, ks[2], ks[3]), [1, 3, 2, 4])
    ipv = permutedims(reshape(ipv, :, mh.head, vs[2], vs[3]), [1, 3, 2, 4])

    #size(ipq) == (hs, q_seq_len, head * batch)
    ipq = reshape(ipq, size(ipq, 1), qs[2], :)
    ipk = reshape(ipk, size(ipk, 1), ks[2], :)
    ipv = reshape(ipv, size(ipv, 1), vs[2], :)

    #wait for batch matmul in Flux
    atten = map(1:size(ipq, 3)) do i
        attention(ipq[:, :, i],
                  ipk[:, :, i],
                  ipv[:, :, i];
                  mask= mask === nothing ? mask : mask[:, :, div(i-1, qs[3])+1]
                  )
    end
    atten = cat(atten...; dims=3) #size(atten) == (hs, q_seq_len, head * batch)
    atten = permutedims(reshape(atten, hs, qs[2], mh.head, qs[3]), [1, 3, 2, 4]) #size(atten) == (hs, head, ql, b)
    atten = reshape(atten, h, :) #size(atten) == (h, ql*b)

    out = mh.oproj(atten)
    reshape(out, :, qs[2], qs[3]) #size(out) == (h, q_seq_len, batch)
end

function (mh::MultiheadAttention)(query::AbstractArray{T, 2},
                                  key::AbstractArray{T, 2},
                                  value::AbstractArray{T, 2};
                                  mask=nothing) where T
    # size(query) == (dims, seq_len)
    # dim = size(query)[1]

    # q_seq_len can != k_seq_len
    #ip = cat(query, key, value; dims=1)
    #ipj = mh.iproj(ip)

    # ipq = ipj[1:h, :] # size(ipq) == (h, seq_len)
    # ipk = ipj[h+1:2h, :]
    # ipv = ipj[2h+1:3h, :]

    # selectdim/view break on gpu
    # ipq = selectdim(ipj, 1, 1:h) # size(ipq) == (h, seq_len)
    # ipk = selectdim(ipj, 1, h+1:2h)
    # ipv = selectdim(ipj, 1, 2h+1:3h)

    # hq = [Tracker.collect(selectdim(ipq, 1, (i-1)*hs+1:i*hs)) for i = 1:mh.head] # head * size(hq[1]) == head * (hs, seq_len)
    # hk = [Tracker.collect(selectdim(ipk, 1, (i-1)*hs+1:i*hs)) for i = 1:mh.head]
    # hv = [Tracker.collect(selectdim(ipv, 1, (i-1)*hs+1:i*hs)) for i = 1:mh.head]

    ipq = mh.iqproj(query)
    ipk = mh.ikproj(key)
    ipv = mh.ivproj(value)

    h = size(ipq)[1] #h == hs * head
    hs = div(h, mh.head)

    hq = [ipq[(i-1)*hs+1:i*hs, :] for i = 1:mh.head] # head * size(hq[1]) == head * (hs, seq_len)
    hk = [ipk[(i-1)*hs+1:i*hs, :] for i = 1:mh.head]
    hv = [ipv[(i-1)*hs+1:i*hs, :] for i = 1:mh.head]


    # size(atten) == (head*hs, seq_len)
    atten = map((q,k,v)->attention(q, k, v; mask=mask, future=mh.future), hq, hk, hv)
    atten = cat(atten...; dims=1)

    mh.oproj(atten)
end

function attention(query::AbstractArray{T, 2},
                   key::AbstractArray{T, 2},
                   value::AbstractArray{T, 2};
                   mask=nothing, future::Bool = false) where T
    # size(query) == (dims, {q,k}_seq_len) == size(key) == size(value)
    # size(score) == (k_seq_len, q_seq_len)
    dk = size(key)[1]
    score = key' * query
    score = score ./ sqrt(dk)

    if mask !== nothing
        @. mask = (1 - mask) * -1e9
        score = score + mask
    end

    if !future
        fmask = fill(convert(T, 1), size(score))
        fmask .-= one(fmask)
        fmask .= -1e9 .* collect(LowerTriangular(fmask))
        fmask = device(fmask)
        score = score + fmask
    end

    score = softmax(score)
    value * score #size(return) == (dims, q_seq_len)
end
