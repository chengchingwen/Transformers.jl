using Flux: @treelike

"""
    TransformerModel(embed::AbstractEmbed, transformers::AbstractTransformer)
    TransformerModel(
                      embed::AbstractEmbed,
                      transformers::AbstractTransformer,
                      classifier
                     )

a structure for putting everything together
"""
struct TransformerModel{E <: AbstractEmbed, T <: AbstractTransformer, C}
    embed::E
    transformers::T
    classifier::C
end

TransformerModel(embed, transformers) = TransformerModel(embed, transformers, identity)

@treelike TransformerModel

"""
    set_classifier(model::TransformerModel, classifier)

return a new TransformerModel whose classifier is set to `classifier`.
"""
set_classifier(model::TransformerModel, classifier) = TransformerModel(model.embed, model.transformers, classifier)

"""
    clear_classifier(model::TransformerModel)

return a new TransformerModel without classifier.
"""
clear_classifier(model::TransformerModel) = TransformerModel(model.embed, model.transformers, identity)

Base.print(io::IO, model::TransformerModel) = Base.summary(io, model)

function recursive_print(io::IO, x, i=1; tabchar = '\t')
    print(io, x)
    print(io, '\n')
end

function recursive_print(io::IO, x::AbstractArray{T}, i=1; tabchar = '\t') where T
    if T == Any || T <: Number
        recursive_print(io, typeof(x), i; tabchar = tabchar)
    else
        recursive_print(io, Tuple(x), i; tabchar = tabchar)
    end
end

function recursive_print(io::IO, x::Union{Tuple, NamedTuple}, i=1; tabchar = '\t')
    print(io, "(\n$(tabchar)")
    for k ∈ keys(x)
        print(io, tabchar^i)
        print(io, "$k => ")
        recursive_print(io, x[k], i+1; tabchar = tabchar)
        print(io, "$(tabchar)")
    end
    print(io, tabchar^(i-1))
    print(io, ")\n")
end

function Base.show(io::IO, model::TransformerModel)
    tabchar = "  "
    print(io, "TransformerModel{")
    print(io, typeof(model.transformers))
    print(io, "}(\n$(tabchar)")
    print(io, "embed = ")
    print(io, model.embed)
    print(io, ",\n$(tabchar)")
    print(io, "transformers = ")
    print(io, model.transformers)
    if model.classifier !== identity
        print(io, ",\n$(tabchar)")
        print(io, "classifier = \n$(tabchar^2)")
        recursive_print(io, model.classifier, 2; tabchar = tabchar)
    else
        print(io, '\n')
    end
    print(io, ")")
end
