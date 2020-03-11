const _get_gpt_config = Dict(
    "OpenAIftlm" => Dict(
        :host => :openai,
        :vocab => 40000,
        :layer => 12,
        :hidden => 768,
        :head => 12,
        :bpe => :bpe,
        :name => "OpenAIftlm",
        :items => (:gpt_model, :bpe, :vocab, :tokenizer)
    )
)

function gpt_init()
    for (model, url, cksum) ∈ zip(
        ("OpenAIftlm",
         ),
        ("https://docs.google.com/uc?export=download&id=1jZ4wSrR7rGXprKz1a3O8PpEQ2o-nnMEA",
         ),
        ("e372ab1f42fff84efc6e97d30d0aadd823706e9b6d4084a8c474ba68c494e22c",
         )
    )
        config = _get_gpt_config[model]
        head = config[:head]
        hidden = config[:hidden]
        layer = config[:layer]
        vocab = config[:vocab]
        bpe = config[:bpe] == :bpe ? " bpe" : ""
        band = config[:host] == :openai ? "OpenAI released " : ""
        register(DataDep(
            "GPT-$model",
            """
            $(band)gpt model with $(layer)-layer, $(head)-head, $(hidden)-hidden, $(vocab)$(bpe) vocabularies
            """,
            url,
            cksum;
            fetch_method = download_gdrive
        ))
    end
end
