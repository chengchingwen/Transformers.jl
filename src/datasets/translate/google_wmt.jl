
googlewmt_init() = register(DataDep(
    "Google-WMT en-de",
    """
    \"\"\"shows in wmt14 of torchtext
    The WMT 2014 English-German dataset, as preprocessed by Google Brain.

    Though this download contains test sets from 2015 and 2016, the train set
    differs slightly from WMT 2015 and 2016 and significantly from WMT 2017.
    \"\"\"

    contain bpe training set and news testset from 2009~2016 (include origin text,
    tokenized, and bpe versions), and also a bpe.32000 and vocab.32000 (merged vocab)
    """,
    "https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8",
    "86f7f6e0bab3259a34712b4c034dc933f406d6735ce03fd6de6b9ccb5191ce2e";
    fetch_method=download_gdrive,
    post_fetch_method=DataDeps.unpack,
))

struct GoogleWMT <: Dataset end

function testfile(::GoogleWMT, year=2014; mode=:bpe)
    !(2009 <= year <= 2016) && error("year shoud be in 2009~2016")
    if mode == :bpe
        en = @datadep_str "Google-WMT en-de/newstest$(year).tok.bpe.32000.en"
        de = @datadep_str "Google-WMT en-de/newstest$(year).tok.bpe.32000.de"
    elseif mode == :tok
        en = @datadep_str "Google-WMT en-de/newstest$(year).tok.en"
        de = @datadep_str "Google-WMT en-de/newstest$(year).tok.de"

    elseif mode == :plain
        en = @datadep_str "Google-WMT en-de/newstest$(year).en"
        de = @datadep_str "Google-WMT en-de/newstest$(year).de"
    else
        error("model should be one of :bpe, :tok, and :plain")
    end
    en, de
end

function trainfile(::GoogleWMT)
    en = datadep"Google-WMT en-de/train.tok.clean.bpe.32000.en"
    de = datadep"Google-WMT en-de/train.tok.clean.bpe.32000.de"
    en, de
end

function get_vocab(::GoogleWMT; mode = :vocab)
    if mode == :vocab
        vf = datadep"Google-WMT en-de/vocab.bpe.32000"
        voc = open(vf) do f
            d = Dict{String, Int}()
            for (i, l) ∈ enumerate(eachline(f))
                d[intern(l)] = i
            end
            d
        end
        return voc
    elseif mode == :bpe
        vf = datadep"Google-WMT en-de/bpe.32000"
        bpe = Bpe(vf; sepsym="@@", have_header=false)
        return bpe
    else
        error("mode should be one of :vocab and :bpe")
    end
end
