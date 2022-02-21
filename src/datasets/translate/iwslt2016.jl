using LightXML

function iwslt2016_init()

    # Helper function for extracting nested archives
    function extract_lang_archive(src, dst, fn)
        # Unpack outer archive and construct path to the archive for the language pair
        unpack(fn)
        archivename = "2016-01"
        innerdir = "$(src)-$(dst)"
        langarchive = joinpath(archivename, "texts", "$(src)", "$(dst)", innerdir * ".tgz")

        # Unpack language pair archive
        unpack(langarchive)
        innerfiles = readdir(innerdir)
        mv.(joinpath.(innerdir, innerfiles), innerfiles)

        for f ∈ innerfiles
             if occursin(".xml", f)
                 clean_xml(f)
             elseif occursin(".tags", f)
                 clean_tag(f)
             end
        end
        rm(innerdir)
        rm(archivename; recursive=true)
    end

    archivehash = "425a3688e0faff00ed4d6d04f1664d1edbd9932e5b17a73680aa81a70f03e2d6"

    message = (src, dst) -> """
    The IWSLT 2016 TED talk translation task

    These are the data sets for the MT tasks of the evaluation campaigns of IWSLT. They are parallel data sets used for building and testing MT systems. They are publicly available through the WIT3 website wit3.fbk.eu, see release: 2016-01.

    Data are crawled from the TED website and carry the respective licensing conditions (for training, tuning and testing MT systems).
    Approximately, for each language pair, training sets include 2,000 talks, 200K sentences and 4M tokens per side, while each dev and test sets 10-15 talks, 1.0K-1.5K sentences and 20K-30K tokens per side. In each edition, the training sets of previous editions are re-used and updated with new talks added to the TED repository in the meanwhile.

    from $(src) to $(dst)
    """

    for lang ∈ ("ar", "cs", "fr", "de")
        register(DataDep(
            "IWSLT2016 $(lang)-en",
            message(lang, "en"),
            "https://drive.google.com/file/d/1l5y6Giag9aRPwGtuZHswh3w5v3qEz8D8/",
            archivehash;
            fetch_method=gdownload,
            post_fetch_method = fn -> extract_lang_archive(lang, "en", fn)
        ))

        register(DataDep(
            "IWSLT2016 en-$(lang)",
            message("en", lang),
            "https://drive.google.com/file/d/1l5y6Giag9aRPwGtuZHswh3w5v3qEz8D8/",
            archivehash;
            fetch_method=gdownload,
            post_fetch_method = fn -> extract_lang_archive("en", lang, fn)
        ))
    end
end

function clean_xml(f)
    nf = replace(f, "xml"=>"txt")
    xdoc = parse_file(f)
    xroot = root(xdoc)
    node = length(xroot["srcset"]) == 0 ? xroot["refset"][1] : xroot["srcset"][1]
    open(nf, "w+") do fw
        for doc in node["doc"]
            for seg in doc["seg"]
                println(fw, content(seg))
            end
        end
    end
    free(xdoc)
    nf
end

function clean_tag(f)
    nf = replace(f, ".tags"=>"")
    open(f) do fr
        open(nf, "w+") do fw
            for l ∈ eachline(fr)
                if !startswith(l, "<")
                    println(fw, l)
                end
            end
        end
    end
    nf
end

struct IWSLT2016 <: Dataset
    src
    ref
    function IWSLT2016(src, ref)
        provided = [:cs, :ar, :fr, :de]
        !(src == :en ? ref ∈ provided : ref == :en && src ∈ provided) && error("language not provided")
        new(src, ref)
    end
end

function trainfile(iw::IWSLT2016)
    p = "$(iw.src)-$(iw.ref)"
    src = @datadep_str "IWSLT2016 $p/train.$(p).$(iw.src)"
    ref = @datadep_str "IWSLT2016 $p/train.$(p).$(iw.ref)"
    src, ref
end

function tunefile(iw::IWSLT2016, dev, year; tedx = false)
    p = "$(iw.src)-$(iw.ref)"
    hX = tedx ? "X" : ""
    catl = dev ? "dev" : "tst"
    srcf = "IWSLT16.TED$(hX).$(catl)$(year).$(p).$(iw.src).txt"
    reff = "IWSLT16.TED$(hX).$(catl)$(year).$(p).$(iw.ref).txt"

    !(srcf ∈ readdir(@datadep_str "IWSLT2016 $p")) && error("""no such file: $srcf,
                                                             only have the following:
                                                             $(join(filter(x->occursin(".$(iw.src).txt", x), readdir(datadep"IWSLT2016 $p/")), "\n"))""")

    src = @datadep_str "IWSLT2016 $p/$srcf"
    ref = @datadep_str "IWSLT2016 $p/$reff"
    src, ref
end

devfile(iw::IWSLT2016, year=2010; tedx = false) = tunefile(iw, true, year; tedx=tedx)
testfile(iw::IWSLT2016, year=2010; tedx = false) = tunefile(iw, false, year; tedx=tedx)


function get_vocab(iw::IWSLT2016; mode = :vocab, min_freq = 7)
    if mode == :vocab
        sf, rf = trainfile(iw)
        vocab = token_freq(sf, rf; min_freq = min_freq)

        return vocab
    else
        error("mode should be one of :vocab")
    end
end
