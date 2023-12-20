using DataStructures
using TextEncodeBase
using TextEncodeBase: AbstractTokenizer, AbstractTokenization, ParentStages,
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
container_f(::T) where T = (container_reducef(T), MutableLinkedList{container_eltype(T)}())

TextEncodeBase.tokenize(tkr::TextTokenizer, p::ParentStages, t::AbstractTokenization, x::Batch) =
    collect(TextEncodeBase.tokenize_procedure!(container_f(x)..., tkr, p, t, x))
