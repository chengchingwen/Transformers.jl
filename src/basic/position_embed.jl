using Flux: @treelike

struct PositionEmbedding
    embedding
end

@treelike PositionEmbedding

function PE(size, pos, i::Int)
    if div(i, 2) == 0
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
    len = size(x)[2]
    selectdim(pe.embedding, 2, 1:len)
end
