using DelimitedFiles

function storycloze_init()
    register(DataDep(
        "StoryClozeTest",
        """
        Story Cloze Test and ROCStories Corpora

        http://cs.rochester.edu/nlp/rocstories/

        Please fill in the form in the url,
        the dataset owner will send the download link to you with email.
        please copy-paste the link after you get the link.

        """,
        "http://cs.rochester.edu/nlp/rocstories/",
        "27eb558d129b922ddb7d5427ee140a21d9d636fbd46c90343fd800ab533f9cb6";
        fetch_method= (dummyremote, localdir) -> begin
            println(dummyremote)
            println("Please paste link you get after recieve the email")
            print("ROCStories winter 2017 set: ")
            rocw_url = readline()
            print("ROCStories spring 2016 set: ")
            rocs_url = readline()
            println("Story Cloze Test Spring 2016 set:")
            print("* val set: ")
            valset_url = readline()
            print("* test set: ")
            testset_url = readline()
            maybegoogle_download(rocw_url, localdir)
            maybegoogle_download(rocs_url, localdir)
            maybegoogle_download(valset_url, localdir)
            maybegoogle_download(testset_url, localdir)
        end
    ))
end

struct StoryCloze <: Dataset end

function testfile(::StoryCloze)
    sets, headers = readdlm(datadep"StoryClozeTest/cloze_test_test__spring2016-cloze_test_ALL_test.csv", ',', String; header=true)

    [selectdim(sets, 2, i) for i = 2:8]
end

function trainfile(::StoryCloze)
    sets, headers = readdlm(datadep"StoryClozeTest/cloze_test_val__spring2016-cloze_test_ALL_val.csv", ',', String; header=true)

    [selectdim(sets, 2, i) for i = 2:8]
end

get_labels(::StoryCloze) = ("1", "2")
