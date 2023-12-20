using TextEncodeBase
using TextEncodeBase: DefaultTokenization, WrappedTokenization, Splittable, SentenceNormalizer,
    ParentStages, WordStage, getvalue

struct PrecompiledNormalizer{T<:AbstractTokenization} <: SentenceNormalizer{T}
    base::T
    precompiled::PrecompiledNorm
end
PrecompiledNormalizer(precompiled) = PrecompiledNormalizer(DefaultTokenization(), precompiled)

TextEncodeBase.normalizer(t::PrecompiledNormalizer) = t.precompiled

struct UnigramTokenization{T <: AbstractTokenization, U <: AbstractUnigram} <: WrappedTokenization{T}
    base::T
    unigram::U
end
UnigramTokenization(unigram::AbstractUnigram) = UnigramTokenization(DefaultTokenization(), unigram)

TextEncodeBase.splittability(::ParentStages, ::UnigramTokenization, ::WordStage) = Splittable()
TextEncodeBase.splitting(::ParentStages, t::UnigramTokenization, w::WordStage) = t.unigram(getvalue(w))
