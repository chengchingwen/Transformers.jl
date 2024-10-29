using Transformers
using FuncPipelines, TextEncodeBase
using TextEncodeBase: nested2batch, nestedcall  
using Flux, StatsBase

conf = Transformers.HuggingFace.load_config("distilbert/distilroberta-base")

tkr = Transformers.HuggingFace.load_tokenizer("distilbert/distilroberta-base")
model = Transformers.HuggingFace.load_model("distilbert/distilroberta-base", :ForMaskedLM)

new_tkr = Transformers.TextEncoders.BertTextEncoder(tkr) do e
	e.process[1:5] |> Pipeline{:masked_position}(nested2batch ∘ nestedcall(isequal("<mask>")), :token) |> e.process[6:end-1] |> PipeGet{(:token, :segment, :attention_mask, :masked_position)}()
end

query = "Paris is the<mask> of France."

input = Transformers.TextEncoders.encode(new_tkr, query)

input_ids = input.masked_position

model_output = model(input)
mask_logits = model_output.logit[:, :, 1]

mask_probabilities = softmax(mask_logits, dims=1)
predicted_token_id = map(argmax, eachcol(mask_probabilities))

predicted_token = Transformers.TextEncoders.decode(new_tkr, predicted_token_id)#[input_ids]