"""
    PositionEmbedding(size::Int, max_len::Int = 1024; trainable::Bool = false)

The position embedding layer. `size` is the number of neuron. `max_len` is the maximum acceptable length of input.
If is not `trainable`, `max_len` will dynamically adjust to the longest input length. If `trainable`, use a random init
embedding value, otherwise use a sin/cos position encoding.
"""
mutable struct PositionEmbedding{F, W <: AbstractArray{F}} <: AbstractBroadcastEmbed{F}
    trainable::Bool
    embedding::W
end

@treelike PositionEmbedding

get_value(e::PositionEmbedding, name::Symbol, xs::NamedTuple) = e(first(xs))

function PE(size, pos, i::Int)
    if rem(i, 2) == 0
        sin(pos/1e4^(i/size))
    else
        cos(pos/1e4^((i-1)/size))
    end
end

function PositionEmbedding(size::Int, max_len::Int = 1024; trainable::Bool = false)
    if trainable
        embedding = param(randn(Float32, size, max_len))
    else
        embedding = Matrix{Float32}(undef, size, max_len)
        for l = 1:max_len
            map!(i->PE(size, l, i), selectdim(embedding, 2, l), 1:size)
        end
    end
    PositionEmbedding(trainable, embedding)
end

(pe::PositionEmbedding)(x::AbstractArray{Int}) = pe(size(x, 1))
(pe::PositionEmbedding{F})(x::AbstractArray{F}) where F = pe(size(x, 2))
function (pe::PositionEmbedding)(len::Int)
    max_len = size(pe.embedding, 2)

    if len > max_len
        if pe.trainable
            error("position embedding length exceeded")
        else
            over = similar(pe.embedding, size(pe.embedding, 1), len)
            copyto!(over, 1, pe.embedding, 1, length(pe.embedding))

            for l = size(pe.embedding, 2)+1:len
                map!(i->PE(size(pe.embedding, 1), l, i), selectdim(over, 2, l), 1:size(pe.embedding, 1))
            end

            pe.embedding = over
        end
    end

    idx = cumsum(fill!(similar(pe.embedding, Int, len), one(Int)))
    gather(pe.embedding, idx)
end

function Base.show(io::IO, pe::PositionEmbedding)
    s, max_len = size(pe.embedding)
    if pe.trainable
        print(io, "PositionEmbedding($(s), max_len=$(max_len))")
    else
        print(io, "PositionEmbedding($(s))")
    end
end
