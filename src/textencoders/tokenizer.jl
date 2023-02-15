using TextEncodeBase
using TextEncodeBase: AbstractTokenizer, AbstractTokenization,  ParentStages,
    Batch, Document, Sentence, DocumentStage, SentenceStage, SubSentenceStage,
    WordStage, SubWordStage, TokenStages, TokenStage

struct TextTokenizer{T <: AbstractTokenization} <: AbstractTokenizer
    tokenization::T
end
TextTokenizer() = TextTokenizer(TextEncodeBase.DefaultTokenization())

TextEncodeBase.tokenization(tkr::TextTokenizer) = tkr.tokenization

container_eltype(::Type{<:Batch{T}}) where T<:Union{SentenceStage, DocumentStage} = Vector{TokenStage}
container_eltype(::Type{<:Batch{Batch{T}}}) where T<:SentenceStage = Vector{Vector{TokenStage}}
container_eltype(::Type{<:Batch{Batch{Batch{T}}}}) where T<:SentenceStage = Vector{Vector{Vector{TokenStage}}}
container_reducef(::Type{<:Batch}) = push!
container_f(::T) where T = (container_reducef(T), container_eltype(T)[])

function TextEncodeBase.tokenize(tkr::TextTokenizer, p::ParentStages, t::AbstractTokenization, x::Batch)
    f, c = container_f(x)
    return TextEncodeBase.tokenize_procedure!(f, c, tkr, p, t, x)
end
