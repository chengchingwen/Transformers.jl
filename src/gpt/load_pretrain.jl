using JSON
using NPZ

using BytePairEncoding
using WordTokenizers

using Flux: loadparams!

"""
The function in the origin gpt code

fixes some issues the spacy tokenizer had on books corpus
also does some whitespace standardization
"""
function text_standardize(text)
    text = replace(text, "—"=>"-")
    text = replace(text, "–"=>"-")
    text = replace(text, "―"=>"-")
    text = replace(text, "…"=>"...")
    text = replace(text, "´"=>"'")
    text = replace(text, r"""(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)"""=>s" \1 ")
    text = replace(text, r"\s*\n\s*"=>" \n ")
    text = replace(text, r"[^\S\n]+"=>" ")
    strip(text)
end


function load_gpt_pretrain_params()
    shapes = JSON.parsefile(joinpath(dirname(@__FILE__), "pretrain/params_shapes.json"))
    offsets = accumulate(+, prod.(shapes))
    params = cat([npzread(joinpath(dirname(@__FILE__), "pretrain/params_$(i).npy")) for i = 0:9]..., dims=1)
    params = [collect(reshape(selectdim(params, 1, a+1:b), s...)) for (a, b, s) in zip([0;offsets[1:end-1]], offsets, shapes)]
    map(params, shapes) do p, s
        l = length(s)
        if l == 3
            reshape(p, s[2:end]...)'
        elseif l == 2
            p'
        else
            p
        end
    end
end

# pm_name = (:pe, :embed, (
#     (:mh.iqproj.W, :mh.ikproj.W, :mh.ivproj.W),
#     (:mh.iqproj.b, :mh.ikproj.b, :mh.ivproj.b),
#     :mh.oproj.W, :mh.oproj.b,
#     :LN1.α, :LN1.β,
#     :pw.din.W, :pw.din.b,
#     :pw.dout.W, :pw.dout.b,
#     :LN2.α, :LN2.β),
#            :xN,
#            )

function load_gpt_pretrain(n::Int=12)
    n > 12 && error("pretrain maximum layer: 12")
    set_tokenizer((x)->nltk_word_tokenize(text_standardize(x)))
    emp = JSON.parsefile(joinpath(dirname(@__FILE__), "pretrain/encoder_bpe_40000.json"))
    vocab = map(first, sort!(collect(emp), by=(x)->x.second))
    push!(vocab, "_start_")
    push!(vocab, "_delimiter_")
    push!(vocab, "_classify_")

    bpe = Bpe(joinpath(dirname(@__FILE__), "pretrain/vocab_40000.bpe"))

    embed = Embed(768, vocab, "<unk>")
    gpt = Gpt(768, 12, 768*4, 12; max_len=512, trainable=true, act=gelu)

    pms = load_gpt_pretrain_params()
    loadparams!(embed, [hcat(pms[2], randn(768, 3))])
    loadparams!(gpt.pe, [pms[1]])
    for i = 1:n
        mhW = pms[12(i-1) + 3]
        mhb = pms[12(i-1) + 4]
        loadparams!(gpt.ts[i].mh.iqproj,[selectdim(mhW, 1, 1:768),
                                         selectdim(mhb, 1, 1:768)])
        loadparams!(gpt.ts[i].mh.ikproj,[selectdim(mhW, 1, 768+1:2*768),
                                         selectdim(mhb, 1, 768+1:2*768)])
        loadparams!(gpt.ts[i].mh.ivproj,[selectdim(mhW, 1, 2*768+1:3*768),
                                         selectdim(mhb, 1, 2*768+1:3*768)])
        loadparams!(gpt.ts[i].mh.oproj,[pms[12(i-1) + 5],
                                        pms[12(i-1) + 6]])
        loadparams!(gpt.ts[i].LN1,[pms[12(i-1) + 7],
                                   pms[12(i-1) + 8]])
        loadparams!(gpt.ts[i].pw.din,[pms[12(i-1) + 9],
                                      pms[12(i-1) + 10]])
        loadparams!(gpt.ts[i].pw.dout,[pms[12(i-1) + 11],
                                       pms[12(i-1) + 12]])
        loadparams!(gpt.ts[i].LN2,[pms[12(i-1) + 13],
                                   pms[12(i-1) + 14]])
    end
    gpt, embed, bpe
end
