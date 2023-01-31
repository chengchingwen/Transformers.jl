using Flux
using StatsBase
using TextEncodeBase
using Transformers
using Transformers.HuggingFace

const textenc = hgf"t5-small:tokenizer"
const model = hgf"t5-small:ForConditionalGeneration"

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
    enc_encoded = encode(textenc, context).token
    enc_input = (; token = enc_encoded)
    dec_encoded = OneHotArray([ lookup(OneHot, textenc.vocab, textenc.padsym) ])
    ids = dec_encoded.onehots
    ends_id = lookup(textenc.vocab, textenc.endsym)
    for i in 1:max_length
        input = (; encoder_input = enc_input, decoder_input = (; token = dec_encoded))
        outputs = model(input)
        logits = @view outputs.logit[:, end, 1]
        probs = temp_softmax(logits)
        new_id = greedy ? Flux.onecold(probs) : top_k_sample(probs)[1]
        push!(ids, new_id)
        new_id == ends_id && break
    end
    return decode(textenc, dec_encoded)
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
