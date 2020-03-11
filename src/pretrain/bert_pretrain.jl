const _get_bert_config = Dict(
    "uncased_L-12_H-768_A-12" => Dict(
        :host => :google,
        :case => :uncased,
        :layer => 12,
        :hidden => 768,
        :head => 12,
        :wwm => false,
        :name => "uncased_L-12_H-768_A-12",
        :items => (:bert_model, :wordpiece, :tokenizer)
    ),
    "uncased_L-24_H-1024_A-16" => Dict(
        :host => :google,
        :case => :uncased,
        :layer => 24,
        :hidden => 1024,
        :head => 16,
        :wwm => false,
        :name => "uncased_L-24_H-1024_A-16",
        :items => (:bert_model, :wordpiece, :tokenizer)
    ),
    "wwm_cased_L-24_H-1024_A-16" => Dict(
        :host => :google,
        :case => :cased,
        :layer => 24,
        :hidden => 1024,
        :head => 16,
        :wwm => true,
        :name => "wwm_cased_L-24_H-1024_A-16",
        :items => (:bert_model, :wordpiece, :tokenizer)
    ),
    "wwm_uncased_L-24_H-1024_A-16" => Dict(
        :host => :google,
        :case => :cased,
        :layer => 24,
        :hidden => 1024,
        :head => 16,
        :wwm => true,
        :name => "wwm_uncased_L-24_H-1024_A-16",
        :items => (:bert_model, :wordpiece, :tokenizer)
    ),
    "multilingual_L-12_H-768_A-12" => Dict(
        :host => :google,
        :case => :multilingual,
        :layer => 12,
        :hidden => 768,
        :head => 12,
        :wwm => false,
        :name => "multilingual_L-12_H-768_A-12",
        :items => (:bert_model, :wordpiece, :tokenizer)
    ),
    "multi_cased_L-12_H-768_A-12" => Dict(
        :host => :google,
        :case => :multi_cased,
        :layer => 12,
        :hidden => 768,
        :head => 12,
        :wwm => false,
        :name => "multi_cased_L-12_H-768_A-12",
        :items => (:bert_model, :wordpiece, :tokenizer)
    ),
    "chinese_L-12_H-768_A-12" => Dict(
        :host => :google,
        :case => :chinese,
        :layer => 12,
        :hidden => 768,
        :head => 12,
        :wwm => false,
        :name => "chinese_L-12_H-768_A-12",
        :items => (:bert_model, :wordpiece, :tokenizer)
    ),
    "cased_L-24_H-1024_A-16" => Dict(
        :host => :google,
        :case => :cased,
        :layer => 24,
        :hidden => 1024,
        :head => 16,
        :wwm => false,
        :name => "cased_L-24_H-1024_A-16",
        :items => (:bert_model, :wordpiece, :tokenizer)
    ),
    "cased_L-12_H-768_A-12" => Dict(
        :host => :google,
        :case => :cased,
        :layer => 12,
        :hidden => 768,
        :head => 12,
        :wwm => false,
        :name => "cased_L-12_H-768_A-12",
        :items => (:bert_model, :wordpiece, :tokenizer)
    ),
    "scibert_scivocab_uncased" => Dict(
        :host => :allenai,
        :case => :cased,
        :layer => 12,
        :hidden => 768,
        :head => 12,
        :wwm => false,
        :name => "scibert_scivocab_uncased",
        :items => (:bert_model, :wordpiece, :tokenizer)
    ),
    "scibert_scivocab_cased" => Dict(
        :host => :allenai,
        :case => :cased,
        :layer => 12,
        :hidden => 768,
        :head => 12,
        :wwm => false,
        :name => "scibert_scivocab_cased",
        :items => (:bert_model, :wordpiece, :tokenizer)
    ),
    "scibert_basevocab_uncased" => Dict(
        :host => :allenai,
        :case => :cased,
        :layer => 12,
        :hidden => 768,
        :head => 12,
        :wwm => false,
        :name => "scibert_basevocab_uncased",
        :items => (:bert_model, :wordpiece, :tokenizer)
    ),
    "scibert_basevocab_cased" => Dict(
        :host => :allenai,
        :case => :cased,
        :layer => 12,
        :hidden => 768,
        :head => 12,
        :wwm => false,
        :name => "scibert_basevocab_cased",
        :items => (:bert_model, :wordpiece, :tokenizer)
    ),
)



function bert_init()
    for (model, url, cksum) ∈ zip(
        ("uncased_L-12_H-768_A-12",
         "uncased_L-24_H-1024_A-16",
         "wwm_cased_L-24_H-1024_A-16",
         "wwm_uncased_L-24_H-1024_A-16",
         "multilingual_L-12_H-768_A-12",
         "multi_cased_L-12_H-768_A-12",
         "chinese_L-12_H-768_A-12",
         "cased_L-24_H-1024_A-16",
         "cased_L-12_H-768_A-12",
         "scibert_scivocab_uncased",
         "scibert_scivocab_cased",
         "scibert_basevocab_uncased",
         "scibert_basevocab_cased",
         ),
        ("https://docs.google.com/uc?export=download&id=1X-p84u7LDEJlhAnDXnyrXbbFo9Y0UBzu",
         "https://docs.google.com/uc?export=download&id=1IR7RWI2ZORzR7JmPSPrUtZk9jdcOm0Na",
         "https://docs.google.com/uc?export=download&id=1sqVrc_PdhYQzdt903yvx7OROtmpjRLUi",
         "https://docs.google.com/uc?export=download&id=1FrpWNCIgVCNGTF-rlU8_3oj1Xp4S94e9",
         "https://docs.google.com/uc?export=download&id=1BJUHaVkpnDL71VlVJS2yRjNF72l1LmWD",
         "https://docs.google.com/uc?export=download&id=1teAmrvOqJfyEa2yYFyNLW7M8JOmg-4sn",
         "https://docs.google.com/uc?export=download&id=14OaijyaNjpK69-Oh7NddIXFlLrgfkf35",
         "https://docs.google.com/uc?export=download&id=1TsaVQHlhOVG905elbHXeXK2qzp-vFe68",
         "https://docs.google.com/uc?export=download&id=1p4mej6xhfpqzLfRuMNkKHJb9ScjXU193",
         "https://docs.google.com/uc?export=download&id=1L0woI2DeNES5OCzgLNBOCRrj5ikOzN7c",
         "https://docs.google.com/uc?export=download&id=1YYy6cH_gQf9rXmnPW9861N0Chlf_ZmTF",
         "https://docs.google.com/uc?export=download&id=16EgAh0uo7pB7aQKCMeI5dfMkCJ1y-RvB",
         "https://docs.google.com/uc?export=download&id=1Htg-qTj03YRQqBgbHxz8KOULBYQGQieP",
         ),
        ("631de5b205c8cb535818ab3a0a2c25c8ac66d841197225496bc26d0ae9fa5702",
         "c696f7fa9c9a774e61d5bff83bd2a0f2263698591be3344400a58c3c88af8e8f",
         "1d77c9128d4262f58aea8d2aa6d21becba241f607168c24e2fc76163d0db7b0a",
         "391b8721471dae54780ea61f1edd286ce2fa67aa9ff47a28807e96b006ef21df",
         "cb2a1b24b4b99434932c647c21baab4ae7867b45053b2a804d8213e9bdc2e04e",
         "ee239ec5fc30ce0c817626dbbf4b8f25119a1b202930f7f1b50ee88b98f26647",
         "94cf78daa24cc3652deec24c786cbe8067fe8dec824d34d7280a0ca9069dc8a1",
         "d08bd9011df02afcf324fc8888304e4868f5bbfca061b4396262903c7d7a7fe0",
         "a02aadcc30c4e49355ebe350b3933dcdcecf7e50af3ea442b71e3c728423ac3a",
         "aa87890fe7c56a30ec4df79cdfabc575448efb6f31e119e6dc0d53150940b1ea",
         "2d6b0e839d5485dc9e5eedcc967671f134308316fe3ee3e9e249cc2aa502f1c4",
         "216afe3542536ac32579f8b81c2b3564c2a8a0033a214274ddacecf215b97100",
         "e5a3b2a692e3971185979c0d8c6882f092380bd580db79a32e72e1b4ec77537f",
         )
    )
        config = _get_bert_config[model]
        case = config[:case]
        head = config[:head]
        hidden = config[:hidden]
        layer = config[:layer]
        wwm = config[:wwm] ? ", whole word masked" : ""
        band = "$(Base.Unicode.titlecase(string(config[:host]))) released "
        register(DataDep(
            "BERT-$model",
            """
            $(band)bert model with $(layer)-layer, $(head)-head, $(hidden)-hidden, $(case)$(wwm)
            """,
            url,
            cksum;
            fetch_method = download_gdrive
        ))
    end
end
