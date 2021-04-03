using BytePairEncoding, JSON, HTTP
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

encoder = JSON.parsefile("./model_vocab/vocab.json")
decoder = map(first, sort!(collect(encoder), by=(x)->x.second))
bpe = Bpe("./model_vocab/bpe.out"; sepsym="", endsym="")

model = hgf"gpt2:lmheadmodel"

function encode_text(text)
  text_ = replace(text, " "=>"Ġ")
  xs = [bpe(text_)...]
  xs[end] = replace(xs[end], "</w>"=>"")
  tokens = []
  for i in 1:length(xs)
    push!(tokens, encoder[xs[i]])
  end
  return tokens .+ 1
end

function temp_softmax(logits; temperature=1.2)
  return (exp.(logits ./temperature)) / sum(exp.(logits./temperature))
end

function top_k_sample(probs; k=10)
  sorted = sort(probs, rev = true)
  indexes = partialsortperm(probs, 1:k, rev=true)
  index = sample(indexes, ProbabilityWeights(sorted[1:k]), 1)
  return index
end

function generate_text(;context="", max_length=50)
  input_= encode_text(context)
  out_tokens = input_
  for i in 1:max_length
    input_ids = reshape(Array{Int64}(out_tokens), (:, 1))
    attention_mask = reshape(Array{Int64}(ones(length(input_ids))), (:,1))
    outputs = model(input_ids; attention_mask=attention_mask,
                              output_attentions=false,
                              output_hidden_states=false,
                              use_cache=false);
    logits = outputs.logits[:, end, 1]
    probs = temp_softmax(logits)
    new_token = top_k_sample(probs)[1]
    push!(out_tokens, new_token)
  end
  return out_tokens
end

function decode_text(text_token_ids)
  tokens = text_token_ids  # ignore the first start token_indices
  text_list = []
  for i in 1:length(tokens)
    push!(text_list, decoder[tokens[i]])
  end
  text = replace(join(text_list), "Ġ"=> " ")
  return text
end

function generate()
  text_token_ids = generate_text(context = "Fruits are very good for "; max_length=100)
  gen_text = decode_text(text_token_ids)
  print("\n\nGenerated Text: ")
  print(gen_text)
end

generate()