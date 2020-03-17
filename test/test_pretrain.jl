#from https://github.com/JuliaText/Embeddings.jl/blob/master/test/runtests.jl
"""
    tempdatadeps(fun)
Run the function and delete all created datadeps afterwards
"""
function tempdatadeps(fun)
    tempdir = mktempdir()
    try
        @info "sending all datadeps to $tempdir"
        withenv("DATADEPS_LOAD_PATH"=>tempdir) do
            fun()
        end
    finally
        try
            @info "removing $tempdir"
            rm(tempdir, recursive=true, force=true)
        catch err
            @warn "Something went wrong with removing tempdir" tempdir exception=err
        end
    end
end

"""
@testset_nokeep_data
Use just like @testset,
but know that it deletes any downloaded data dependencies when it is done.
"""
macro testset_nokeep_data(name, expr)
    quote
        tempdatadeps() do
            @testset $name $expr
        end
    end |> esc
end

@testset "Pretrain" begin
  using Transformers.Pretrain
  using DataDeps
  using BytePairEncoding
  ENV["DATADEPS_ALWAYS_ACCEPT"] = true
  ENV["DATADEPS_NO_STANDARD_LOAD_PATH"] = true
  ENV["DATADEPS_PROGRESS_UPDATE_PERIOD"] = "Inf"

  @test_nowarn pretrains()
  model_list = map(l->(join([l[1],l[3]], "-"), l[1], l[3]),
                   pretrains().content[1].rows[2:end])

  for (model_str, type, name) ∈ model_list
    @testset_nokeep_data "$model_str" begin
      GC.gc()
      @test Transformers.Pretrain.parse_model(model_str)[1] == type
      @test Transformers.Pretrain.parse_model(model_str)[2] == name

      if type == "gpt"
        @test_nowarn Transformers.Pretrain.@datadep_str "$(uppercase(type))-$name/$(name).npbson"
      else
        @test_nowarn Transformers.Pretrain.@datadep_str "$(uppercase(type))-$name/$(name).tfbson"
      end
      @test_nowarn x = Transformers.Pretrain.@pretrain_str model_str
    end
  end
end
