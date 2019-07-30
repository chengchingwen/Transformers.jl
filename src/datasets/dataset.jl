abstract type Dataset end

abstract type Mode end
struct Train <: Mode end
struct Dev <: Mode end
struct Test <: Mode end

function dataset( ::Type{M}, d::D, args...; kwargs...) where M <: Mode where D <: Dataset
    ds = datafile(M, d, args...; kwargs...)
    rds = reader.(ds)
    rds
end

datafile(::Type{Train}, d::D, args...; kwargs...) where D <: Dataset = trainfile(d, args...; kwargs...)
datafile(::Type{Dev}, d::D, args...; kwargs...) where D <: Dataset = devfile(d, args...; kwargs...)
datafile(::Type{Test}, d::D, args...; kwargs...) where D <: Dataset = testfile(d, args...; kwargs...)

trainfile(d, args...; kwargs...) = (println("Trainset not found"); nothing)
devfile(d, args...; kwargs...) = (println("Devset not found"); nothing)
testfile(d, args...; kwargs...) = (println("Testset not found"); nothing)

function get_channels(::Type{T}, n; buffer_size=0) where T
    Tuple([Channel{T}(buffer_size) for i = 1:n])
end

function reader(file::AbstractString)
    ch = Channel{String}(0)
    task = @async begin
      open(file) do f
        foreach(x->put!(ch, x), eachline(f))
      end
    end
    bind(ch, task)
    ch
end

function reader(iter)
    ch = Channel{String}(0)
    task = @async foreach(x->put!(ch, x), iter)
    bind(ch, task)
    ch
end

function batched(xs)
    s = length(xs)
    sx = s != 0 ? length(xs[1]) : 0
    res = [Vector{typeof(xs[1][i])}() for i = 1:sx]
    for x ∈ xs
        for (i, xi) ∈ enumerate(x)
            push!(res[i], xi)
        end
    end
    res
end

function get_batch(c::Channel, n=1)
    res = Vector(undef, n)
    for (i, x) ∈ enumerate(c)
        res[i] = x
        i >= n && break
    end
    isassigned(res, n) ? batched(res) : nothing
end

function get_batch(cs::Container{C}, n=1) where C <: Channel
    res = Vector(undef, n)
    for (i, xs) ∈ enumerate(zip(cs...))
        res[i] = xs
        i >= n && break
    end
    isassigned(res, n) ? batched(res) : nothing
end

get_vocab(::D, args...; kwargs...) where D <: Dataset = (println("No prebuild vocab"); nothing)

get_labels(::D, args...; kwargs...) where D <: Dataset = (println("Labels unknown"); nothing)


function token_freq(files...; vocab::Dict{String, Int} = Dict{String,Int}(), min_freq::Int = 3)
    for f ∈ files
        open(f) do fd
            for line ∈ eachline(fd)
                for token ∈ tokenize(line)
                    token = intern(token)
                    vocab[token] = get(vocab, token, 0) + 1
                end
            end
        end
    end

    for key ∈ keys(vocab)
        if vocab[key] < min_freq
            delete!(vocab, key)
        end
    end
    vocab
end
