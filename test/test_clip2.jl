# CLIP
# Get CLIP model config from huggingface
using Revise
using Transformers
using TextEncodeBase
using Transformers.HuggingFace


clip_model_name = "openai/clip-vit-large-patch14"
clip_config = Transformers.load_config(clip_model_name)
embeddings = HuggingFace.HGFCLIPTextEmbeddings(clip_config.text_config)
state = Transformers.load_state(clip_model_name)
HuggingFace.load_state!(embeddings, state.text_model.embeddings)

tkr = hgf"openai/clip-vit-large-patch14:tokenizer"

# for Diffusers, token vectors must return 77 element vector
tkr = Transformers.Basic.TransformerTextEncoder(tkr) do enc
    Pipelines(enc.process[1:5]) |> 
    Pipeline{:trunc_tok}(TextEncodeBase.trunc_or_pad(enc.trunc, enc.padsym), :tok) |>
    Pipelines(enc.process[7:end])
end

e = encode(tkr, ["a photo of an astronaut riding a horse on mars", ""])
s = reinterpret(Int32, e.input.tok)
res = embeddings(s)
size(res) # 768, 77, 2