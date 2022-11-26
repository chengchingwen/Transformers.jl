using Flux
using StatsBase
using TextEncodeBase
using Transformers
using Transformers.Basic
using Transformers.GenerativePreTrain
using Transformers.Pretrain
using Transformers.HuggingFace

const textenc = hgf"t5-small:tokenizer"
const model = todevice(hgf"t5-small:ForConditionalGeneration")

function temp_softmax(logits; temperature=1.2)
    return softmax(logits ./ temperature)
end

function top_k_sample(probs; k=10)
    sorted = sort(probs, rev = true)
    indexes = partialsortperm(probs, 1:k, rev=true)
    index = sample(indexes, ProbabilityWeights(sorted[1:k]), 1)
    return index
end

function generate_text(context=""; max_length=50, greedy = true)
    enc = encode(textenc, [context])
    dec_tokens = [textenc.padsym]
    for i in 1:max_length
        data = lookup(textenc.vocab, dec_tokens)
        outputs = model(enc.input.tok, data)
        logits = @view outputs.logits[:, end, 1]
        probs = temp_softmax(logits)
        pred = greedy ? Flux.onecold(probs) : top_k_sample(probs)[1]
        new_token = lookup(textenc.vocab, pred)
        push!(dec_tokens, new_token)
        new_token == textenc.endsym && break
    end
    return dec_tokens
end

function generate(prompt; max_length = 100, greedy = true)
    tokens = generate_text(prompt; max_length, greedy)
    if tokens[end] == textenc.endsym
        tokens = tokens[2:end-1]
    else
        tokens = tokens[2:end]
    end
    gen_text = replace(join(tokens), '▁'=>' ')
    println("\nGenerated Text: \n")
    println(gen_text)
    return gen_text
end
