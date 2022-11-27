using TextEncodeBase
using TextEncodeBase: DefaultTokenization, WrappedTokenization, Splittable, ParentStages, WordStage, getvalue

struct WordPieceTokenization{T<:AbstractTokenization} <: WrappedTokenization{T}
    base::T
    wordpiece::WordPiece
end
WordPieceTokenization(wordpiece::WordPiece) = WordPieceTokenization(DefaultTokenization(), wordpiece)

TextEncodeBase.splittability(::ParentStages, ::WordPieceTokenization, ::WordStage) = Splittable()
TextEncodeBase.splitting(::ParentStages, t::WordPieceTokenization, w::WordStage) = t.wordpiece(getvalue(w))
