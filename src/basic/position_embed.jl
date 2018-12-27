using Flux: @treelike

mutable struct PositionEmbedding
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

function PositionEmbedding(size::Int, max_len::Int = 1024)
    embedding = Matrix{Float64}(undef, size, max_len)
    for l = 1:max_len
        map!(i->PE(size, l, i), selectdim(embedding, 2, l), 1:size)
    end
    device(PositionEmbedding(embedding))
end


function (pe::PositionEmbedding)(x)
    len = size(x, 2)

    if len > size(pe.embedding, 2)
        over = Matrix{Float64}(undef, size(pe.embedding, 1), len)
        selectdim(over, 2, 1:size(pe.embedding, 2)) .= pe.embedding

        for l = size(pe.embedding, 2)+1:len
            map!(i->PE(size(pe.embedding, 1), l, i), selectdim(over, 2, l), 1:size(pe.embedding, 1))
        end

        pe.embedding = device(over)
    end
    selectdim(pe.embedding, 2, 1:len)
end

Base.show(io::IO, pe::PositionEmbedding) = print(io, "PositionEmbedding($(size(pe.embedding)[1]))")
