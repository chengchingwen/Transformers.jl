using LightXML

function iwslt2016_init()
    for (lang, checksum) ∈ zip(("ar", "cs", "fr", "de"), (("0e7dd1c836f66f0e68c45c3dea6312dd6a1f2e8c93bcf03982f842644319ff4c",
                                                           "bf10a15077b4d25cc3e8272e61429d6e85f75a0a55f180a10d01abfcdb3debc9"),
                                                          ("d6d5a4767f6afc96c59630583d0cfe2b23ba24a48a33c16b7fdb76017ee62ab3",
                                                           "f38dcf1407afa224324dbed02424fa2c30042d10a5cc387d02df1369eec3f68e"),
                                                          ("132bc5524c1f7500aadb84a4e05a0c3dd15cc5b527d4d4af402fd98582299231",
                                                           "b70aca9675966fcbdfb8349086848638f6711e47c669f5654971859d10266398"),
                                                          ("7e21dd345e9192180f36d7816f84a77eafd6b85e45432d90d0970f06e8c772ea",
                                                           "13c037b8a5dce7fb6199eeedc6b0460c0c75082db8eeda21902acb373ba9ba14")))
        register(DataDep(
            "IWSLT2016 $(lang)-en",
            """
            The IWSLT 2016 TED talk translation task

            These are the data sets for the MT tasks of the evaluation campaigns of IWSLT. They are parallel data sets used for building and testing MT systems. They are publicly available through the WIT3 website wit3.fbk.eu, see release: 2016-01.

            Data are crawled from the TED website and carry the respective licensing conditions (for training, tuning and testing MT systems).
            Approximately, for each language pair, training sets include 2,000 talks, 200K sentences and 4M tokens per side, while each dev and test sets 10-15 talks, 1.0K-1.5K sentences and 20K-30K tokens per side. In each edition, the training sets of previous editions are re-used and updated with new talks added to the TED repository in the meanwhile.

            from $(lang) to en
            """,
            "https://wit3.fbk.eu/archive/2016-01//texts/$lang/en/$(lang)-en.tgz",
            checksum[1];
            post_fetch_method = fn -> begin
                unpack(fn)
                innerdir = "$(lang)-en"
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
            end
        ))

        register(DataDep(
            "IWSLT2016 en-$(lang)",
            """
            The IWSLT 2016 TED talk translation task

            These are the data sets for the MT tasks of the evaluation campaigns of IWSLT. They are parallel data sets used for building and testing MT systems. They are publicly available through the WIT3 website wit3.fbk.eu, see release: 2016-01.

            Data are crawled from the TED website and carry the respective licensing conditions (for training, tuning and testing MT systems).
            Approximately, for each language pair, training sets include 2,000 talks, 200K sentences and 4M tokens per side, while each dev and test sets 10-15 talks, 1.0K-1.5K sentences and 20K-30K tokens per side. In each edition, the training sets of previous editions are re-used and updated with new talks added to the TED repository in the meanwhile.

            from en to $(lang)
            """,
            "https://wit3.fbk.eu/archive/2016-01//texts/en/$lang/en-$(lang).tgz",
            checksum[2];
            post_fetch_method = fn -> begin
                unpack(fn)
                innerdir = "en-$(lang)"
                innerfiles = readdir(innerdir)
                mv.(joinpath.(innerdir, innerfiles), innerfiles)

                for f ∈ innerfiles
                     if occursin(".xml", f)
                         clean_xml(f)
                     elseif occursin(".tag", f)
                         clean_tag(f)
                     end
                end
                rm(innerdir)
            end
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
                                                             $(join(filter(x->occursin(".en.txt", x), readdir(datadep"IWSLT2016 fr-en/")), "\n"))""")

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
