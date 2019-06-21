# turn a openAI finetune-transformer-lm npy to bson

using JSON
using ZipFile

using Flux: loadparams!

using BytePairEncoding

iszip(s) = endswith(s, ".zip")
isnpbson(s) = endswith(s, ".npbson")

function openAInpy2bson(path;
                        raw=false,
                        saveto="./",
                        startsym="_start_",
                        delisym="_delimiter_",
                        clfsym="_classify_",
                        unksym="<unk>")
  if iszip(path)
    filename = basename(path)[1:end-4]
    data = ZipFile.Reader(path)
  else
    filename = basename(isdirpath(path) ? path[1:end-1] : path)
    data = path
  end

  tokenizer = x -> nltk_word_tokenize(text_standardize(x))
  weights, bpe, raw_vocab = readnpzfolder(data)

  if raw
    bsonname = normpath(joinpath(saveto, filename * ".npbson"))
    BSON.@save bsonname weights bpe raw_vocab tokenizer
  else
    bsonname = normpath(joinpath(saveto, filename * ".bson"))
    gpt_model, vocab = load_gpt_from_npbson(weights, raw_vocab;
                                            startsym=startsym,
                                            delisym=delisym,
                                            clfsym=clfsym,
                                            unksym=unksym,
                                            )

    BSON.@save bsonname gpt_model bpe vocab tokenizer
  end
end

zipname(z::ZipFile.Reader) = z.files[1].name
zipfile(z::ZipFile.Reader, name) = (idx = findfirst(zf->isequal(name)(zf.name), z.files)) !== nothing ? z.files[idx] : nothing



readnpz(path) = error("readnpz require NPZ.jl installed. run `Pkg.add(\"NPZ\"); using NPZ`")

@init @require NPZ="15e1cf62-19b3-5cfa-8e77-841668bca605" begin
  import .NPZ

  function readnpz(dir)
    shapes = JSON.parsefile(joinpath(dir, "params_shapes.json"))
    offsets = accumulate(+, prod.(shapes))
    shapes = map(s -> length(s) > 1 ? (s[end], s[end-1]) : s, shapes)
    params = cat([npzread(joinpath(dir, "params_$(i).npy")) for i = 0:9]..., dims=1)
    params = [collect(reshape(selectdim(params, 1, a+1:b), s...)) for (a, b, s) in zip([0;offsets[1:end-1]], offsets, shapes)]
    params
  end

  function readnpz(z::ZipFile.Reader)
    shapes = JSON.parse(zipfile(z, joinpath(zipname(z), "params_shapes.json")))
    offsets = accumulate(+, prod.(shapes))
    shapes = map(s -> length(s) > 1 ? (s[end], s[end-1]) : s, shapes)
    params = cat([NPZ.npzreadarray(zipfile(z, joinpath(zipname(z), "params_$(i).npy"))) for i = 0:9]..., dims=1)
    params = [collect(reshape(selectdim(params, 1, a+1:b), s...)) for (a, b, s) in zip([0;offsets[1:end-1]], offsets, shapes)]
    params
  end
end

function readnpzfolder(dir)
  emp = JSON.parsefile(joinpath(dir, "encoder_bpe_40000.json"))
  vocab = map(first, sort!(collect(emp), by=(x)->x.second))
  bpe = Bpe(joinpath(dir, "vocab_40000.bpe"))
  weights = readnpz(dir)

  weights, bpe, vocab
end

function readnpzfolder(z::ZipFile.Reader)
  emp = JSON.parse(zipfile(z, joinpath(zipname(z), "encoder_bpe_40000.json")))
  vocab = map(first, sort!(collect(emp), by=(x)->x.second))
  bpe = mktemp(
    (fn, f)-> begin
      write(f, read(zipfile(z, joinpath(zipname(z), "vocab_40000.bpe"))))
      close(f)
      Bpe(fn)
    end
  )
  weights = readnpz(z)
  weights, bpe, vocab
end


load_gpt_from_npbson(path::AbstractString) = (@assert isnpbson(path); load_gpt_from_npbson(BSON.load(path)))
load_gpt_from_npbson(bson) = load_gpt_from_npbson(bson[:weights], bson[:raw_vocab])
function load_gpt_from_npbson(weights, raw_vocab;
                              startsym="_start_",
                              delisym="_delimiter_",
                              clfsym="_classify_",
                              unksym="<unk>")
    vocab = copy(raw_vocab)

    push!(vocab, startsym)
    push!(vocab, delisym)
    push!(vocab, clfsym)

    vocab = Vocabulary(vocab, unksym)
    embed = Embed(768, length(vocab))
    pe = PositionEmbedding(768, 512; trainable=true)

    gpt = Gpt(768, 12, 768*4, 12; act=gelu, pdrop=0.1, attn_pdrop=0.1)


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

    loadparams!(embed, [hcat(weights[2],
                             randn(768, 3) .* 0.02)])
    loadparams!(pe, [weights[1]])
    for i = 1:12
        mhW = weights[12(i-1) + 3]
        mhb = weights[12(i-1) + 4]
        loadparams!(gpt.ts[i].mh.iqproj,[selectdim(mhW, 1, 1:768),
                                         selectdim(mhb, 1, 1:768)])
        loadparams!(gpt.ts[i].mh.ikproj,[selectdim(mhW, 1, 768+1:2*768),
                                         selectdim(mhb, 1, 768+1:2*768)])
        loadparams!(gpt.ts[i].mh.ivproj,[selectdim(mhW, 1, 2*768+1:3*768),
                                         selectdim(mhb, 1, 2*768+1:3*768)])
        loadparams!(gpt.ts[i].mh.oproj,[weights[12(i-1) + 5],
                                        weights[12(i-1) + 6]])
        loadparams!(gpt.ts[i].mhn,[weights[12(i-1) + 7],
                                   weights[12(i-1) + 8]])
        loadparams!(gpt.ts[i].pw.din,[weights[12(i-1) + 9],
                                      weights[12(i-1) + 10]])
        loadparams!(gpt.ts[i].pw.dout,[weights[12(i-1) + 11],
                                       weights[12(i-1) + 12]])
        loadparams!(gpt.ts[i].pwn,[weights[12(i-1) + 13],
                                   weights[12(i-1) + 14]])
    end

    ce = CompositeEmbedding(tok=embed, pe=pe)
    TransformerModel(ce, gpt), vocab
end



