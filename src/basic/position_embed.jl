using Flux: @treelike
using Flux.Tracker: data

broadcast_add(e, pe) = e .+ pe
function broadcast_add(e::ThreeDimArray{T}, pe) where T
    #for Flux gpu issue 530 https://github.com/FluxML/Flux.jl/issues/530
    s = size(e)
    reshape(reshape(e, :, s[end]) .+ reshape(pe, :, 1), s)
end

mutable struct PositionEmbedding
    trainable::Bool
    embedding
end

@treelike PositionEmbedding

function PE(size, pos, i::Int)
    if rem(i, 2) == 0
        sin(pos/1e4^(i/size))
    else
        cos(pos/1e4^((i-1)/size))
    end
end

function PositionEmbedding(size::Int, max_len::Int = 1024; trainable::Bool = false)
    if trainable
        embedding = param(randn(size, max_len))
    else
        embedding = Matrix{Float64}(undef, size, max_len)
        for l = 1:max_len
            map!(i->PE(size, l, i), selectdim(embedding, 2, l), 1:size)
        end
    end
    PositionEmbedding(trainable, embedding)
end

function (pe::PositionEmbedding)(x)
    len = size(x, 2)
    max_len = size(pe.embedding, 2)

    if len > max_len
        if pe.trainable
            error("position embedding length exceeded")
        else
            over = Matrix{eltype(data(pe.embedding))}(undef, size(pe.embedding, 1), len)
            selectdim(over, 2, 1:size(pe.embedding, 2)) .= pe.embedding

            for l = size(pe.embedding, 2)+1:len
                map!(i->PE(size(pe.embedding, 1), l, i), selectdim(over, 2, l), 1:size(pe.embedding, 1))
            end

            pe.embedding = device(over)
        end
    end
    pe.embedding[:, 1:len]
end

function Base.show(io::IO, pe::PositionEmbedding)
    s, max_len = size(pe.embedding)
    if pe.trainable
        print(io, "PositionEmbedding($(s), max_len=$(max_len))")
    else
        print(io, "PositionEmbedding($(s))")
    end
end
