using TextEncodeBase
using TextEncodeBase: trunc_and_pad, trunc_or_pad
using NeuralAttentionlib: LengthMask, RevLengthMask

check_vocab(vocab::Vocab, word) = findfirst(==(word), vocab.list) !== nothing

# used for grouping tokenized tokens into correct format for SequenceTemplate
grouping_sentence(x::AbstractVector{<:AbstractString}) = x # single sentence
grouping_sentence(x::AbstractVector{<:AbstractVector{<:AbstractString}}) = map(Base.vect, x) # batch of sentence
grouping_sentence(x::AbstractVector{<:AbstractVector{<:AbstractVector{<:AbstractString}}}) = x

getlengths(maxlength) = TextEncodeBase.FixRest(getlengths, maxlength)
getlengths(x::AbstractArray, maxlength) = _getlength(maxlength, x)
getlengths(x::AbstractArray{<:AbstractVector}, maxlength) = map(getlengths(maxlength), x)
function getlengths(x::AbstractArray{>:AbstractArray}, maxlength)
    aoa, aov = TextEncodeBase.allany(Base.Fix2(isa, AbstractArray), x)
    if aoa
        map(getlengths(maxlength), x)
    elseif aov
        _getlength(maxlength, x)
    else
        error("Input array is mixing array and non-array elements")
    end
end

function _getlength(maxlength, x)
    len = length(x)
    isnothing(maxlength) && return len
    return min(len, maxlength)
end

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

function get_mask_func(trunc, pad_end)
    maskf = pad_end == :head ? RevLengthMask : LengthMask
    return maskf ∘ getlengths(trunc)
end
