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

trainfile(d, args...; kwargs...) = error("Trainset not found")
devfile(d, args...; kwargs...) = error("Devset not found")
testfile(d, args...; kwargs...) = error("Testset not found")


trainset(d::D, args...) where D <: Dataset = dataset(Train, d, args...)

function reader(file)
    ch = Channel{String}(0)
    task = @async foreach(x->put!(ch, x), eachline(open(file)))
    bind(ch, task)
    ch
end

function get_batch(c::Channel, n=1)
    res = Vector()
    for (i, x) ∈ enumerate(c)
        push!(res, x)
        i >= n && break
    end
    res
end

function get_batch(cs::NTuple{N}{Channel}, n=1) where N
    res = Vector()
    for (i, xs) ∈ enumerate(zip(cs...))
        push!(res, xs)
        i >= n && break
    end
    res
end

get_vocab(::D, args...; kwargs...) where D <: Dataset = error("No prebuild vocab")
