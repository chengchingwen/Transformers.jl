using HuggingFaceApi

using HuggingFaceApi: PYTORCH_WEIGHTS_NAME, CONFIG_NAME

const VOCAB_FILE = "vocab.txt"

# Slow tokenizers used to be saved in three separated files
const SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
const ADDED_TOKENS_FILE = "added_tokens.json"
const TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
# Fast tokenizers (provided by HuggingFace tokenizer's library) can be saved in a single file
const FULL_TOKENIZER_FILE = "tokenizer.json"

hgf_model_config_url(model_name; revision = "main") =
    HuggingFaceURL(model_name, CONFIG_NAME; repo_type = nothing, revision = something(revision, "main"))
hgf_model_weight_url(model_name; revision = "main") =
    HuggingFaceURL(model_name, PYTORCH_WEIGHTS_NAME; repo_type = nothing, revision = something(revision, "main"))
hgf_vocab_url(model_name; revision = "main") =
    HuggingFaceURL(model_name, VOCAB_FILE; repo_type = nothing, revision = something(revision, "main"))
hgf_tokenizer_special_tokens_map_url(model_name; revision = "main") =
    HuggingFaceURL(model_name, SPECIAL_TOKENS_MAP_FILE; repo_type = nothing, revision = something(revision, "main"))
hgf_tokenizer_added_token_url(model_name; revision = "main") =
    HuggingFaceURL(model_name, ADDED_TOKENS_FILE; repo_type = nothing, revision = something(revision, "main"))
hgf_tokenizer_config_url(model_name; revision = "main") =
    HuggingFaceURL(model_name, TOKENIZER_CONFIG_FILE; repo_type = nothing, revision = something(revision, "main"))
hgf_tokenizer_url(model_name; revision = "main") =
    HuggingFaceURL(model_name, FULL_TOKENIZER_FILE; repo_type = nothing, revision = something(revision, "main"))

function _hgf_download(
    hgfurl::HuggingFaceURL; local_files_only::Bool = false, cache::Bool = true,
    auth_token = HuggingFaceApi.get_token(), kw...
)
    return hf_hub_download(
        hgfurl.repo_id, hgfurl.filename;
        repo_type = hgfurl.repo_type, revision = hgfurl.revision,
        auth_token, local_files_only, cache
    )
end

hgf_model_config(model_name; revision = "main", kw...) =
    _hgf_download(hgf_model_config_url(model_name; revision = something(revision, "main")); kw...)
hgf_model_weight(model_name; revision = "main", kw...) =
    _hgf_download(hgf_model_weight_url(model_name; revision = something(revision, "main")); kw...)
hgf_vocab(model_name; revision = "main", kw...) =
    _hgf_download(hgf_vocab_url(model_name; revision = something(revision, "main")); kw...)
hgf_tokenizer_special_tokens_map(model_name; revision = "main", kw...) =
    _hgf_download(hgf_tokenizer_special_tokens_map_url(model_name; revision = something(revision, "main")); kw...)
hgf_tokenizer_added_token(model_name; revision = "main", kw...) =
    _hgf_download(hgf_tokenizer_added_token_url(model_name; revision = something(revision, "main")); kw...)
hgf_tokenizer_config(model_name; revision = "main", kw...) =
    _hgf_download(hgf_tokenizer_config_url(model_name; revision = something(revision, "main")); kw...)
hgf_tokenizer(model_name; revision = "main", kw...) =
    _hgf_download(hgf_tokenizer_url(model_name; revision = something(revision, "main")); kw...)
