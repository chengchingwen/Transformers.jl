using HuggingFaceApi

using HuggingFaceApi: PYTORCH_WEIGHTS_NAME, CONFIG_NAME

const PYTORCH_WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
const SAFETENSOR_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
const SAFETENSOR_WEIGHTS_NAME = "model.safetensors"

const VOCAB_FILE = "vocab.txt"
const VOCAB_JSON_FILE = "vocab.json"
const MERGES_FILE = "merges.txt"

# Slow tokenizers used to be saved in three separated files
const SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
const ADDED_TOKENS_FILE = "added_tokens.json"
const TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
# Fast tokenizers (provided by HuggingFace tokenizer's library) can be saved in a single file
const FULL_TOKENIZER_FILE = "tokenizer.json"

hgf_file_url(model_name, file_name; revision = "main") =
    HuggingFaceURL(model_name, file_name; repo_type = nothing, revision = something(revision, "main"))

hgf_model_config_url(model_name; revision = "main") = hgf_file_url(model_name, CONFIG_NAME; revision)
hgf_model_weight_url(model_name; revision = "main") = hgf_file_url(model_name, PYTORCH_WEIGHTS_NAME; revision)
hgf_model_weight_index_url(model_name; revision = "main") = hgf_file_url(model_name, PYTORCH_WEIGHTS_INDEX_NAME; revision)
hgf_vocab_url(model_name; revision = "main") = hgf_file_url(model_name, VOCAB_FILE; revision)
hgf_vocab_json_url(model_name; revision = "main") = hgf_file_url(model_name, VOCAB_JSON_FILE; revision)
hgf_tokenizer_special_tokens_map_url(model_name; revision = "main") = hgf_file_url(model_name, SPECIAL_TOKENS_MAP_FILE; revision)
hgf_tokenizer_added_token_url(model_name; revision = "main") = hgf_file_url(model_name, ADDED_TOKENS_FILE; revision)
hgf_tokenizer_config_url(model_name; revision = "main") = hgf_file_url(model_name, TOKENIZER_CONFIG_FILE; revision)
hgf_tokenizer_url(model_name; revision = "main") = hgf_file_url(model_name, FULL_TOKENIZER_FILE; revision)
hgf_merges_url(model_name; revision = "main") = hgf_file_url(model_name, MERGES_FILE; revision)

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

hgf_file(model_name, file_name; revision = "main", kws...) = _hgf_download(hgf_file_url(model_name, file_name; revision); kws...)

hgf_model_config(model_name; kws...) = hgf_file(model_name, CONFIG_NAME; kws...)
hgf_model_weight(model_name; kws...) = hgf_file(model_name, PYTORCH_WEIGHTS_NAME; kws...)
hgf_model_weight_index(model_name; kws...) = hgf_file(model_name, PYTORCH_WEIGHTS_INDEX_NAME; kws...)
hgf_model_safetensor_weight(model_name; kws...) = hgf_file(model_name, SAFETENSOR_WEIGHTS_NAME; kws...)
hgf_model_safetensor_weight_index(model_name; kws...) = hgf_file(model_name, SAFETENSOR_WEIGHTS_INDEX_NAME; kws...)
hgf_vocab(model_name; kws...) = hgf_file(model_name, VOCAB_FILE; kws...)
hgf_vocab_json(model_name; kws...) = hgf_file(model_name, VOCAB_JSON_FILE; kws...)
hgf_tokenizer_special_tokens_map(model_name; kws...) = hgf_file(model_name, SPECIAL_TOKENS_MAP_FILE; kws...)
hgf_tokenizer_added_token(model_name; kws...) = hgf_file(model_name, ADDED_TOKENS_FILE; kws...)
hgf_tokenizer_config(model_name; kws...) = hgf_file(model_name, TOKENIZER_CONFIG_FILE; kws...)
hgf_tokenizer(model_name; kws...) = hgf_file(model_name, FULL_TOKENIZER_FILE; kws...)
hgf_merges(model_name; kws...) = hgf_file(model_name, MERGES_FILE; kws...)
