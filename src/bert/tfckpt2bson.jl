# turn a tf bert format to bson

using JSON
using ZipFile

using Flux: loadparams!

function named2tokenizer(name)
  if occursin("uncased", name)
    return bert_uncased_tokenizer
  else
    return bert_cased_tokenizer
  end
end

"""
    tfckpt2bson(path;
                raw=false,
                saveto="./",
                confname = "bert_config.json",
                ckptname = "bert_model.ckpt",
                vocabname = "vocab.txt")

turn google released bert format into BSON file. Set `raw` to `true` to remain the origin data format in bson.
"""
function tfckpt2bson(path; raw=false, saveto="./", confname = "bert_config.json", ckptname = "bert_model.ckpt", vocabname = "vocab.txt")
  if iszip(path)
    data = ZipFile.Reader(path)
  else
    data = path
  end

  config, weights, vocab = readckptfolder(data; confname=confname, ckptname=ckptname, vocabname=vocabname)

  iszip(path) && close(data)

  if raw
    #saveto tfbson (raw julia data)
    bsonname = normpath(joinpath(saveto, config["filename"] * ".tfbson"))
    BSON.@save bsonname config weights vocab
  else
    #turn raw julia data to transformer model type
    bert_model = load_bert_from_tfbson(config, weights)
    wordpiece = WordPiece(vocab)
    tokenizer = named2tokenizer(config["filename"])
    bsonname = normpath(joinpath(saveto, config["filename"] * ".bson"))
    BSON.@save bsonname bert_model wordpiece tokenizer
  end

  bsonname
end

"loading tensorflow checkpoint file into julia Dict"
readckpt(path) = error("readckpt require TensorFlow.jl installed. run `Pkg.add(\"TensorFlow\"); using TensorFlow`")

@init @require TensorFlow="1d978283-2c37-5f34-9a8e-e9c0ece82495" begin
  import .TensorFlow
  #should be changed to use c api once the patch is included
  function readckpt(path)
    weights = Dict{String, Array}()
    TensorFlow.init()
    ckpt = TensorFlow.pywrap_tensorflow.x.NewCheckpointReader(path)
    shapes = ckpt.get_variable_to_shape_map()

    for (name, shape) ∈ shapes
      weight = ckpt.get_tensor(name)
      if length(shape) == 2 && name != "cls/seq_relationship/output_weights"
        weight = collect(weight')
      end
      weights[name] = weight
    end

    weights
  end
end

function readckptfolder(z::ZipFile.Reader; confname = "bert_config.json", ckptname = "bert_model.ckpt", vocabname = "vocab.txt")
  (confile = findfile(z, confname)) === nothing && error("config file $confname not found")
  findfile(z, ckptname*".meta") === nothing && error("ckpt file $ckptname not found")
  (vocabfile = findfile(z, vocabname)) === nothing && error("vocab file $vocabname not found")

  dir = zipname(z)
  filename = basename(isdirpath(dir) ? dir[1:end-1] : dir)

  config = JSON.parse(confile)
  config["filename"] = filename
  vocab = readlines(vocabfile)

  weights = mktempdir(
    dir -> begin
      #dump ckpt to tmp
      for fidx ∈ findall(zf->startswith(zf.name, joinpath(zipname(z), ckptname)), z.files)
        zf = z.files[fidx]
        zfn = basename(zf.name)
        f = open(joinpath(dir, zfn), "w+")
        buffer = Vector{UInt8}(undef, zf.uncompressedsize)
        write(f, read!(zf, buffer))
        close(f)
      end

      readckpt(joinpath(dir, ckptname))
    end
  )

  config, weights, vocab
end

function readckptfolder(dir; confname = "bert_config.json", ckptname = "bert_model.ckpt", vocabname = "vocab.txt")
  files = readdir(dir)

  confname ∉ files && error("config file $confname not found")
  ckptname*".meta" ∉ files && error("ckpt file $ckptname not found")
  vocabname ∉ files && error("vocab file $vocabname not found")
  filename = basename(isdirpath(dir) ? dir[1:end-1] : dir)

  config = JSON.parsefile(joinpath(dir, confname))
  config["filename"] = filename
  vocab = readlines(open(joinpath(dir, vocabname)))
  weights = readckpt(joinpath(dir, ckptname))
  config, weights, vocab
end

function get_activation(act_string)
    if act_string == "gelu"
        gelu
    elseif act_string == "relu"
        relu
    elseif act_string == "tanh"
        tanh
    elseif act_string == "linear"
        identity
    else
        throw(DomainError(act_string, "activation support: linear, gelu, relu, tanh"))
    end
end

_create_classifier(;args...) = args.data

load_bert_from_tfbson(path::AbstractString) = (@assert istfbson(path); load_bert_from_tfbson(BSON.load(path)))
load_bert_from_tfbson(bson) = load_bert_from_tfbson(bson[:config], bson[:weights])
function load_bert_from_tfbson(config, weights)
    #init bert model possible component
    bert = Bert(
        config["hidden_size"],
        config["num_attention_heads"],
        config["intermediate_size"],
        config["num_hidden_layers"];
        act = get_activation(config["hidden_act"]),
        pdrop = config["hidden_dropout_prob"],
        attn_pdrop = config["attention_probs_dropout_prob"]
    )

    embedding = Dict{Symbol, Any}()

    tok_emb = Embed(
        config["hidden_size"],
        config["vocab_size"]
    )

    seg_emb = Embed(
        config["hidden_size"],
        config["type_vocab_size"]
    )

    posi_emb = PositionEmbedding(
        config["hidden_size"],
        config["max_position_embeddings"];
        trainable = true
    )

    emb_post = Positionwise(
        LayerNorm(
             config["hidden_size"]
        ),
        Dropout(
            config["hidden_dropout_prob"]
        )
    )

    classifier = Dict{Symbol, Any}()

    pooler = Dense(
        config["hidden_size"],
        config["hidden_size"],
        tanh
    )

    masklm = (
        transform = Chain(
            Dense(
                config["hidden_size"],
                config["hidden_size"],
                get_activation(config["hidden_act"])
            ),
            LayerNorm(
                config["hidden_size"]
            )
        ),
        output_bias = param(randn(
            Float32,
            config["vocab_size"]
        ))
    )

    nextsentence = Chain(
        Dense(
            config["hidden_size"],
            2
        ),
        logsoftmax
    )

    #tf namespace handling
    vnames = keys(weights)
    bert_weights = filter(name->occursin("layer", name), vnames)
    embeddings_weights = filter(name->occursin("embeddings", name), vnames)
    pooler_weights = filter(name->occursin("pooler", name), vnames)
    masklm_weights = filter(name->occursin("cls/predictions", name), vnames)
    nextsent_weights = filter(name->occursin("cls/seq_relationship", name), vnames)

    for i = 1:config["num_hidden_layers"]
        li_weights = filter(name->occursin("layer_$(i-1)/", name), bert_weights)
        for k ∈ li_weights
            if occursin("layer_$(i-1)/attention", k)
                if occursin("self/key/kernel", k)
                    loadparams!(bert[i].mh.ikproj.W, [weights[k]])
                elseif occursin("self/key/bias", k)
                    loadparams!(bert[i].mh.ikproj.b, [weights[k]])
                elseif occursin("self/query/kernel", k)
                    loadparams!(bert[i].mh.iqproj.W, [weights[k]])
                elseif occursin("self/query/bias", k)
                    loadparams!(bert[i].mh.iqproj.b, [weights[k]])
                elseif occursin("self/value/kernel", k)
                    loadparams!(bert[i].mh.ivproj.W, [weights[k]])
                elseif occursin("self/value/bias", k)
                    loadparams!(bert[i].mh.ivproj.b, [weights[k]])
                elseif occursin("output/LayerNorm/gamma", k)
                    loadparams!(bert[i].mhn.diag.α, [weights[k]])
                elseif occursin("output/LayerNorm/beta", k)
                    loadparams!(bert[i].mhn.diag.β, [weights[k]])
                elseif occursin("output/dense/kernel", k)
                    loadparams!(bert[i].mh.oproj.W, [weights[k]])
                elseif occursin("output/dense/bias", k)
                    loadparams!(bert[i].mh.oproj.b, [weights[k]])
                else
                    @warn "unknown variable: $k"
                end
            elseif occursin("layer_$(i-1)/intermediate", k)
                if occursin("kernel", k)
                    loadparams!(bert[i].pw.din.W, [weights[k]])
                elseif occursin("bias", k)
                    loadparams!(bert[i].pw.din.b, [weights[k]])
                else
                    @warn "unknown variable: $k"
                end
            elseif occursin("layer_$(i-1)/output", k)
                if occursin("output/LayerNorm/gamma", k)
                    loadparams!(bert[i].pwn.diag.α, [weights[k]])
                elseif occursin("output/LayerNorm/beta", k)
                    loadparams!(bert[i].pwn.diag.β, [weights[k]])
                elseif occursin("output/dense/kernel", k)
                    loadparams!(bert[i].pw.dout.W, [weights[k]])
                elseif occursin("output/dense/bias", k)
                    loadparams!(bert[i].pw.dout.b, [weights[k]])
                else
                    @warn "unknown variable: $k"
                end
            else
                @warn "unknown variable: $k"
            end
        end
    end

    for k ∈ embeddings_weights
        if occursin("LayerNorm/gamma", k)
            loadparams!(emb_post[1].diag.α, [weights[k]])
            embedding[:postprocessor] = emb_post
        elseif occursin("LayerNorm/beta", k)
            loadparams!(emb_post[1].diag.β, [weights[k]])
        elseif occursin("word_embeddings", k)
            loadparams!(tok_emb.embedding, [weights[k]])
            embedding[:tok] = tok_emb
        elseif occursin("position_embeddings", k)
            loadparams!(posi_emb.embedding, [weights[k]])
            embedding[:pe] = posi_emb
        elseif occursin("token_type_embeddings", k)
            loadparams!(seg_emb.embedding, [weights[k]])
            embedding[:segment] = seg_emb
        else
            @warn "unknown variable: $k"
        end
    end

    for k ∈ pooler_weights
        if occursin("dense/kernel", k)
            loadparams!(pooler.W, [weights[k]])
        elseif occursin("dense/bias", k)
            loadparams!(pooler.b, [weights[k]])
        else
            @warn "unknown variable: $k"
        end
    end

    if !isempty(pooler_weights)
        classifier[:pooler] = pooler
    end


    for k ∈ masklm_weights
        if occursin("predictions/output_bias", k)
            loadparams!(masklm.output_bias, [weights[k]])
        elseif occursin("predictions/transform/dense/kernel", k)
            loadparams!(masklm.transform[1].W, [weights[k]])
        elseif occursin("predictions/transform/dense/bias", k)
            loadparams!(masklm.transform[1].b, [weights[k]])
        elseif occursin("predictions/transform/LayerNorm/gamma", k)
            loadparams!(masklm.transform[2].diag.α, [weights[k]])
        elseif occursin("predictions/transform/LayerNorm/beta", k)
            loadparams!(masklm.transform[2].diag.β, [weights[k]])
        else
            @warn "unknown variable: $k"
        end
    end

    if !isempty(masklm_weights)
        classifier[:masklm] = masklm
    end

    for k ∈ nextsent_weights
        if occursin("seq_relationship/output_weights", k)
            loadparams!(nextsentence[1].W, [weights[k]])
        elseif occursin("seq_relationship/output_bias", k)
            loadparams!(nextsentence[1].b, [weights[k]])
        else
            @warn "unknown variable: $k"
        end
    end

    if !isempty(nextsent_weights)
        classifier[:nextsentence] = nextsentence
    end

    if Set(vnames) != union(bert_weights, embeddings_weights, pooler_weights, masklm_weights, nextsent_weights)
        @warn "some unkown variable not load"
    end

    embed = CompositeEmbedding(;embedding...)
    cls = _create_classifier(; classifier...)

    TransformerModel(embed, bert, cls)
end

