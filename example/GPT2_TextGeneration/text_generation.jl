using Flux
using StatsBase
using TextEncodeBase
using Transformers
using Transformers.HuggingFace

const textenc = hgf"gpt2:tokenizer"
const model = hgf"gpt2:lmheadmodel"

function temp_softmax(logits; temperature=1.2)
    return softmax(logits ./ temperature)
end

function top_k_sample(probs; k=10)
    sorted = sort(probs, rev = true)
    indexes = partialsortperm(probs, 1:k, rev=true)
    index = sample(indexes, ProbabilityWeights(sorted[1:k]), 1)
    return index
end

function generate_text(context=""; max_length=50)
    encoded = encode(textenc, context).token
    ids = encoded.onehots
    new_ids = ids[0:-1]
    ends_id = lookup(textenc.vocab, textenc.endsym)
    for i in 1:max_length
        input = (; token = OneHotArray(ids))
        outputs = model(input)
        logits = @view outputs.logit[:, end, 1]
        probs = temp_softmax(logits)
        new_id = top_k_sample(probs)[1]
        push!(ids, new_id)
        new_id == ends_id && break
    end
    return decode(textenc, OneHotArray(ids))
end

function generate(prompt, max_length)
    text_token = generate_text(prompt; max_length=max_length)
    gen_text = join(text_token)
    print("\n\nGenerated Text: ")
    println(gen_text)
end

# generate("My name is Thomas and my main", 100)
