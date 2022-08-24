using Flux
using StatsBase
using TextEncodeBase
using Transformers
using Transformers.Basic
using Transformers.GenerativePreTrain
using Transformers.Pretrain
using Transformers.HuggingFace

const textenc = GPT2TextEncoder(hgf"gpt2:tokenizer") do e
    GenerativePreTrain.gpt2_default_preprocess(
        ; trunc = e.trunc, startsym = e.startsym, endsym = nothing, padsym = e.padsym,
        trunc_end = :head, pad_end = :head,
    )
end
const model = todevice(hgf"gpt2:lmheadmodel")

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
    tokens = TextEncodeBase.tokenize(textenc, [context])
    for i in 1:max_length
        data = lookup(textenc, TextEncodeBase.process(textenc, tokens))
        outputs = model(data.input.tok; output_attentions=false,
                        output_hidden_states=false,
                        use_cache=false)
        logits = @view outputs.logits[:, end, 1]
        probs = temp_softmax(logits)
        new_token = lookup(textenc.vocab, top_k_sample(probs)[1])
        push!(tokens[], TextEncodeBase.Token(new_token))
        new_token == "<|endoftext|>" && break
    end
    return map(TextEncodeBase.getvalue, tokens[])
end

function generate(prompt, max_length)
    text_token = generate_text(prompt; max_length=max_length)
    ids = lookup(textenc.vocab, text_token)
    gen_text = join(TextEncodeBase.decode(textenc, ids))
    print("\n\nGenerated Text: ")
    println(gen_text)
end

generate( "Fruits are very good for ", 100)
