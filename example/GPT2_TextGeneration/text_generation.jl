using BytePairEncoding, JSON, HTTP
using Flux
using StatsBase
using Transformers
using Transformers.Basic
using Transformers.GenerativePreTrain
using Transformers.Pretrain
using Transformers.HuggingFace

isdir("model_vocab")||mkdir("model_vocab")

# This is a temporary fix, will be updated once Tokenizer API is ready
isfile("model_vocab/bpe.out") || HTTP.download("https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt", "./model_vocab/bpe.out")
isfile("model_vocab/vocab.json") || HTTP.download("https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json", "./model_vocab/vocab.json")

labels = map(x->x[1], sort!(collect(pairs(JSON.parsefile("./model_vocab/vocab.json"))), by=x->x[2]))
encoder = Vocabulary(labels, "<|endoftext|>")
bpe = ByteLevelBPE("./model_vocab/bpe.out")

model = hgf"gpt2:lmheadmodel"

function encode_text(text)
  xs = ["<|endoftext|>"; bpe(text)]
  return encoder(xs)
end

function temp_softmax(logits; temperature=1.2)
  return softmax(logits ./ temperature)
end

function top_k_sample(probs; k=10)
  sorted = sort(probs, rev = true)
  indexes = partialsortperm(probs, 1:k, rev=true)
  index = sample(indexes, ProbabilityWeights(sorted[1:k]), 1)
  return index
end

function generate_text(;context="", max_length=50)
  input_ = encode_text(context)
  for i in 1:max_length
    input_ids = reshape(@view(input_[:]), :, 1)
    outputs = model(input_ids; output_attentions=false,
                    output_hidden_states=false,
                    use_cache=false)
    logits = @view outputs.logits[:, end, 1]
    probs = temp_softmax(logits)
    new_token = top_k_sample(probs)[1]
    push!(input_, new_token)
  end
  return input_
end

function decode_text(text_token_ids)
  text_list = decode(encoder, text_token_ids)
  text = BytePairEncoding.UnMap(bpe.codemap)(join(text_list))
  return text
end

function generate(prompt, max_length)
  text_token_ids = generate_text(context = prompt; max_length=max_length)
  gen_text = decode_text(text_token_ids)
  print("\n\nGenerated Text: ")
  println(gen_text)
end

generate( "Fruits are very good for ", 100)
