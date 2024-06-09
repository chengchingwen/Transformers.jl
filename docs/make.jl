using Transformers
using Documenter

DocMeta.setdocmeta!(Transformers, :DocTestSetup, :(using Transformers); recursive=true)

makedocs(;
    modules=[Transformers],
    authors="chengchingwen and contributors",
    repo="https://github.com/chengchingwen/Transformers.jl/blob/{commit}{path}#{line}",
    sitename="Transformers.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://chengchingwen.github.io/Transformers.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Get Started" => "getstarted.md",
        "Tutorial" => "tutorial.md",
        "Layers" => "layers.md",
        "TextEncoders" => "textencoders.md",
        "HuggingFace" => [
            "User Interface" => "huggingface.md",
            "Add New Models" => "huggingface_dev.md",
        ],
        "API Reference" => "api_ref.md",
        "ChangeLogs" => "changelog.md",
    ],
)

deploydocs(;
    repo="github.com/chengchingwen/Transformers.jl",
)
