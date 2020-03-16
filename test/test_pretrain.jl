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
    @testset_nokeep_data "$model" begin
      GC.gc()
      @test Pretrain.parse_model(model_str)[1] == type
      @test Pretrain.parse_model(model_str)[2] == name

      @test_nowarn Pretrain.@datadep_str "$(uppercase(type))-$name/$(name).tfbson"
      @test_nowarn x = Pretrain.@pretrain_str model
    end
  end
end
