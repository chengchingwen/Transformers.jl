using Flux: @treelike

"""
    TransformerModel(embed::AbstractEmbed, transformers::AbstractTransformer, classifer)
    TransformerModel(embed::AbstractEmbed, transformers::AbstractTransformer)

a structure for put everything together
"""
struct TransformerModel{E <: AbstractEmbed, T <: AbstractTransformer, C}
    embed::E
    transformers::T
    classifer::C
end

TransformerModel(embed, transformers) = TransformerModel(embed, transformers, identity)

@treelike TransformerModel

function Base.show(io::IO, model::TransformerModel)
    print(io, "TransformerModel{")
    print(io, typeof(model.transformers))
    print(io, "}(")
    print(io, model.embed)
    print(io, ", ")
    print(io, model.transformers)
    if model.classifer !== identity
        print(io, ", ")
        print(io, model.classifer)
    end
    print(io, ")")
end
