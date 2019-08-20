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
  ENV["DATADEPS_PROGRESS_UPDATE_PERIOD"] = "Inf" #should be able remove one day
  ENV["DATADEP_PROGRESS_UPDATE_PERIOD"] = "Inf"
  
  @test_nowarn pretrains()
  model_list = map(l->join([l[1],l[2]], "-"), pretrains().content[1].rows[2:end])

  for model ∈ model_list
    @testset_nokeep_data "$model" begin
      GC.gc()
      if startswith(lowercase(model), "bert") && model[6:end] != "uncased_L-12_H-768_A-12"
        model_name = model[6:end]
        @test_nowarn Transformers.Pretrain.@datadep_str "BERT-$model_name/$model_name.tfbson"
      else
        @test_nowarn x = Transformers.Pretrain.@pretrain_str model
      end
    end
  end

end
