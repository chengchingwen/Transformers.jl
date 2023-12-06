using TextEncodeBase
using TextEncodeBase: trunc_and_pad, trunc_or_pad, nested2batch, peek_sequence_sample_type
using NeuralAttentionlib: LengthMask, RevLengthMask

string_getvalue(x::TextEncodeBase.TokenStage) = String(getvalue(x))::String

# strip with length
function string_strip(f, s::AbstractString; start = typemax(Int), stop = typemax(Int))
    start_count = 0
    for c in Iterators.take(s, start)
        f(c) || break
        start_count += 1
    end
    stop_count = 0
    for c in Iterators.take(Iterators.reverse(s), stop)
        f(c) || break
        stop_count += 1
    end
    return chop(s; head = start_count, tail = stop_count)
end
string_strip(char::Char, s::AbstractString; start = typemax(Int), stop = typemax(Int)) = string_strip(isequal(char), s; start, stop)

# check word is inside the vocab or not
check_vocab(vocab::Vocab, word) = findfirst(==(word), vocab.list) !== nothing

# used for grouping tokenized tokens into correct format for SequenceTemplate
grouping_sentence(x::AbstractVector) = x
grouping_sentence(x::AbstractVector{<:AbstractVector{<:AbstractString}}) = map(Base.vect, x) # batch of sentence

# get length of each sentences
getlengths(maxlength) = TextEncodeBase.FixRest(getlengths, maxlength)
function getlengths(x, maxlength)
    lens = _getlengths(x, maxlength)
    if lens isa Integer
        lens = [lens]
    end
    return nested2batch(lens)
end

_getlengths(maxlength) = TextEncodeBase.FixRest(_getlengths, maxlength)

function _getlengths(x::AbstractArray, maxlength)
    stype = peek_sequence_sample_type(x)
    if stype == TextEncodeBase.SingleSample
        return __getlength(maxlength, x)
    elseif stype >= TextEncodeBase.UnknownSample
        return TextEncodeBase.@elementmap x _getlengths(x, maxlength)
        # return map(_getlengths(maxlength), x)
    else
        error("Input array is mixing array and non-array elements")
    end
end

function __getlength(maxlength, x)
    len = length(x)
    isnothing(maxlength) && return len
    return min(len, maxlength)
end

# function factory for truncation / padding
get_trunc_pad_func(fixedsize, trunc, trunc_end, pad_end) =
    TextEncodeBase.FixRest(get_trunc_pad_func, fixedsize, trunc, trunc_end, pad_end)
function get_trunc_pad_func(padsym, fixedsize, trunc, trunc_end, pad_end)
    if fixedsize
        @assert !isnothing(trunc) "`fixedsize=true` but `trunc` is not set."
        truncf = trunc_or_pad
    else
        truncf = trunc_and_pad
    end
    return truncf(trunc, padsym, trunc_end, pad_end)
end

# function factory for length mask
function get_mask_func(trunc, pad_end)
    maskf = pad_end == :head ? RevLengthMask : LengthMask
    return maskf ∘ getlengths(trunc)
end
