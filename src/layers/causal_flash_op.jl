using CUDA
@inline function compute_shmem_size(d, Bs)
    return (Bs * d * 3 + 4 * d + Bs * Bs) * sizeof(Float32)
end

"""
    setMaxShmem(shmem)

Set the maximum shared memory size for the current device to `shmem` KB.
"""
function setMaxShmem(shmem)
    kernel = cufunction(flash_attention_kernel, NTuple{4, CuDeviceArray{Float16, 4, 1}})
    return CUDA.cuFuncSetAttribute(kernel.fun,
                                   CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                   shmem * 1024)
end

function _checkbounds(Q, K, V)
    sQ, sK, sV = size(Q), size(K), size(V)
    sK != sV && throw(DimensionMismatch("K and V must have the same shape"))
    sQ[3:4] != sK[3:4] != sV[3:4] &&
        throw(DimensionMismatch("Q, K and V must have the same batch size and head size"))
    return sQ[1] != sK[2] != sV[2] &&
           throw(DimensionMismatch("Q, K and V must have the same hidden dimension"))
end

@inline function mod1_pow2(x, y)
    r = x & (y - 1)
    return ifelse(r == 0, y, r)
end


function causal_flash_attention_kernel(Q, K, V, O)
    d = size(K, 1)
    power = trailing_zeros(d)
    tx = threadIdx().x
    Bs = blockDim().x # assume Br == Bc
    col = (blockIdx().x - 1) * Bs + tx
    # skip computation if col < row

    # acllocate shared memory
    T = eltype(Q)
    shmem_offset = 0
    q = CuDynamicSharedArray(T, (Bs + 2, d), shmem_offset) # pad 2 rows to avoid bank conflicts
    shmem_offset += sizeof(q)
    o = CuDynamicSharedArray(T, (Bs + 2, d), shmem_offset) # pad 2 row to avoid bank conflicts
    shmem_offset += sizeof(o)
    k = CuDynamicSharedArray(T, (d, Bs), shmem_offset) # pad 2 rows to avoid bank conflicts
    shmem_offset += sizeof(k)
    s = CuDynamicSharedArray(T, (Bs, Bs), shmem_offset)

    # load Q to shared memory, note that this is done only once
    Q_offset = d * Bs * (blockIdx().x - 1) +
               stride(Q, 3) * (blockIdx().y - 1) +
               stride(Q, 4) * (blockIdx().z - 1)
    K_offset = stride(K, 3) * (blockIdx().y - 1) + stride(K, 4) * (blockIdx().z - 1)

    for i in 0:(d - 1)
        idx = i * Bs + tx
        row = mod1_pow2(idx, d)
        col = (idx - row) >> power + 1
        @inbounds q[col, row] = Q[idx + Q_offset]
        @inbounds o[idx] = zero(T)
        @inbounds k[idx] = K[idx + K_offset]
    end

    sync_threads()

    # initialize lseᵢ and mᵢ
    lseᵢ = -typemax(T)
    mᵢ = -typemax(T)

    # the inner loop is serial
    for _ in 1:cld(size(K, 2), Bs)  # iterate over Bs elements in sequence
        # initialize mᵢⱼ
        mᵢⱼ = lseᵢ

        # compute s=Q^TK
        # s = (Bs, Bs)
        #inf_block = true
        for n in 1:Bs
            if Q_offset + tx < K_offset + n
                s[tx, n] = -Inf
                continue
            end
            #inf_block = false

            tmp = zero(T)
            for m in 1:d
                @inbounds tmp = CUDA.fma(q[tx, m], k[m, n], tmp)
            end
            s[tx, n] = tmp
            @inbounds mᵢⱼ = max(mᵢⱼ, s[tx, n])
        end
        #inf_block && return nothing

        sync_threads()

        # compute P̃ᵢⱼ and lᵢⱼ
        lᵢⱼ = zero(T)
        for n in 1:Bs
            @inbounds tmp = exp(s[tx, n] - mᵢⱼ)
            @inbounds s[tx, n] = tmp
            lᵢⱼ += tmp
        end

        # Load V to shared memory, which shares the same memory with k
        for i in 0:(d - 1)
            idx = i * Bs + tx
            row = mod1_pow2(idx, d)
            col = (idx - row) >> power + 1
            @inbounds k[row, col] = V[idx + K_offset]
        end

        sync_threads()

        # update o
        for m in 1:d
            tmp = o[tx, m] * exp(mᵢ - mᵢⱼ)
            for n in 1:Bs
                @inbounds tmp = CUDA.fma(s[tx, n], k[m, n], tmp) # k[m, n] * s[n, tx]
            end
            @inbounds o[tx, m] = tmp
        end

        mᵢ = mᵢⱼ
        lseᵢ = mᵢⱼ + log(exp(lseᵢ - mᵢⱼ) + lᵢⱼ)

        K_offset += Bs * d

        # update k
        for i in 0:(d - 1)
            idx = i * Bs + tx
            @inbounds k[idx] = K[idx + K_offset]
        end
        sync_threads()
    end

    for m in 1:d
        @inbounds o[tx, m] = o[tx, m] * exp(mᵢ - lseᵢ)
    end
    sync_threads()

    # write to O
    for i in 0:(d - 1)
        idx = i * Bs + tx
        row = mod1_pow2(idx, d)
        col = (idx - row) >> power + 1
        @inbounds O[idx + Q_offset] = o[col, row]
    end

    return nothing
end

function causal_flash_attention(Q::CuArray{T, 4}, K::CuArray{T, 4}, V::CuArray{T, 4}) where {T}
    _checkbounds(Q, K, V)
    O = similar(Q)
    kernel = @cuda launch=false causal_flash_attention_kernel(Q, K, V, O)
    d, N, H, B = size(Q)
    get_shmem = Base.Fix1(compute_shmem_size, d)
    config = launch_configuration(kernel.fun; shmem=get_shmem, max_threads=256)

    Bs = min(N, config.threads)
    threads = (Bs, 1, 1)
    blocks = (cld(N, Bs), H, B)
    shmem = get_shmem(Bs)

    kernel(Q, K, V, O; threads=threads, blocks=blocks, shmem=shmem)
    return O
end

function causal_flash_attention(n_heads::Int, Q, K, V)
    @assert ndims(Q) == ndims(K) == ndims(V) == 3 "Q, K, and V should be of size (d*h, n, b)"
    Q_fa, K_fa, V_fa = Transformers_to_Flash(n_heads, Q), Transformers_to_Flash(n_heads, K), Transformers_to_Flash(n_heads, V)
    O_fa = causal_flash_attention(Q_fa, K_fa, V_fa)
    Flash_to_Transformers(O_fa)
end
function Transformers_to_Flash(n_heads::Int, arr)
    d = Int(size(arr, 1) /  n_heads)
    N, B = size(arr, 2), size(arr, 3)
    arr_4d = reshape(arr, d, n_heads, N, B)
    perm =(1, 3, 2, 4)
    permutedims(arr_4d, perm)
end

function Flash_to_Transformers(arr)
    arr = permutedims(arr, (1,3,2,4))
    N, B = size(arr, 3), size(arr, 4)
    (hidden_state=reshape(arr, :, N, B),)
end


struct CausalFlashMultiheadQKVAttenOp{F} <: AbstractAttenOp
    head::Int
    p::F
end
CausalFlashMultiheadQKVAttenOp(head) = CausalFlashMultiheadQKVAttenOp(head, nothing)
NeuralAttentionlib.get_attention_func(::CausalFlashMultiheadQKVAttenOp) = causal_flash_attention
NeuralAttentionlib.get_attention_func_args(op::CausalFlashMultiheadQKVAttenOp, q, k, v, mask = nothing) = (op.head, q, k, v)
argument_names(::CausalFlashMultiheadQKVAttenOp) = (:hidden_state, :attention_mask)
apply_on_namedtuple(op::CausalFlashMultiheadQKVAttenOp, nt::NamedTuple) = apply_attention_op(op, nt)
