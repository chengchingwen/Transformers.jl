using Flux: @treelike

struct TransformerModel{E <: AbstractEmbed, T <: AbstractTransformer, C}
    embed::E
    transfomers::T
    classifer::C
end

@treelike TransformerModel


