using JSON3

"""
	_gettype(s)
	_gettype(s, name)

	Julia type of the tensor from the string name
"""
function _gettype(s::AbstractString, name="")
	s == "F16" && return(Float16)
	s == "F32" && return(Float32)
	s == "F64" && return(Float64)
	s == "B" && return(Bool)
	s == "U8" && return(UInt8)
	s == "I8" && return(Int8)
	s == "I16" && return(Int16)
	s == "I32" && return(Int32)
	s == "I64" && return(Int64)
	s == "BF16" && error("BFloat16 is not supported")
	name = isempty(name) ? name : " of the tensor "*name
	error("unknown type $(s)", name)
end

_byteoftype(::Type{T}) where {T<:Union{Bool, UInt8, Int8}} = 1
_byteoftype(::Type{T}) where {T<:Union{Int16, Float16}} = 2
_byteoftype(::Type{T}) where {T<:Union{Int32, Float32}} = 4
_byteoftype(::Type{T}) where {T<:Union{Int64, Float64}} = 8

"""
	readtensor!(fio::IO, header::Dict, name::Symbol, header_length; seek_to_start = true)
	readtensor!(fio::IO, T, shape, start, stop, name="", header_length; seek_to_start = true)

	reads tensor `name` from the file `fio`. 
	`seek_to_start = true` means that seek(fio, start) will be called to ensure that reading 
	starts from correct position 
"""
function readtensor!(fio::IO, header::JSON3.Object, name::Symbol, header_length; seek_to_start = true)
	entry = header[name]
	T = _gettype(entry[:dtype], name)
	start = entry[:data_offsets][1] + header_length
	stop = entry[:data_offsets][2] + header_length
	shape = tuple(entry[:shape]...)
	readtensor!(fio, T, shape, start, stop, name; seek_to_start)
end

function readtensor!(fio::IO, T::Type, shape::NTuple{N,<:Integer}, start::Integer, stop::Integer, name=""; seek_to_start = true) where {N}
	seek_to_start && seek(fio, start)
	n = stop - start
	if _byteoftype(T)*prod(shape) != n
		s = isempty(name) ? "" : "of tensor "*name
		error("length of the stored data",s," does not corresponds to shape of the tensor")
	end
	x = Array{T,length(shape)}(undef, reverse(shape)...)
	read!(fio, x)
	if length(shape) == 2
		x = transpose(x)
	end
	length(shape) > 2 && warn("higher dimensional tensor $(name) untested")
 	return(x)
end

function names_without_metadata(header)
	filter(s -> s !== Symbol("__metadata__"), collect(keys(header)))
end

"""
	starts_of_tensors(header)

	return a sorted list of pairs (name_of_tensor, start)
"""
function starts_of_tensors(header)
	ks = names_without_metadata(header)
	starts = map(ks) do k 
		k => header[k][:data_offsets][1]
	end
	sort!(starts, lt = (i,j) -> i[2] < j[2])
	return(starts)
end

"""
	is_continuous(header, starts = starts_of_tensors(header))

	return true if tensors in header are correctly aligned and can be read sequentially (which they should)
"""
function is_continuous(header, starts = starts_of_tensors(header))
	i = 0 
	for (k, start) in starts 
		start != i && return(false)
		i = header[k]["data_offsets"][2]
	end
	return(true)
end



"""
	header, header_length = load_header(fio::IO)

	loads the header of a stream containing safetensor
"""
function load_header(fio::IO)
	seek(fio, 0)
	n = read(fio, Int64) # first read the length of the header
	s = read(fio, n) # then read the header
	header = JSON3.read(s)
	return(header, 8 + n)
end

function load_safetensors(fio::IO, header, tensors, header_length; seek_to_start = true)
	Dict(map(k -> String(k) => readtensor!(fio, header, k, header_length; seek_to_start), tensors))
end

function load_safetensors(filename::AbstractString)
	open(filename,"r") do fio
		header, header_length = load_header(fio)
		starts = starts_of_tensors(header)
		seek_to_start = !is_continuous(header, starts)
		tensors = first.(starts)
		load_safetensors(fio, header, tensors, header_length; seek_to_start)
	end
end
