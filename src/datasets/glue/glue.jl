module GLUE
using DataDeps

using ..Datasets: Dataset, get_channels
import ..Datasets: testfile, devfile, trainfile, get_labels

export CoLA, SST, MRPC, QQP, STS, MNLI, SNLI, QNLI, RTE, WNLI, Diagnostic

function __init__()
    cola_init()
    sst_init()
    mrpc_init()
    qqp_init()
    sts_init()
    mnli_init()
    snli_init()
    qnli_init()
    rte_init()
    wnli_init()
    diagnostic_init()
end

include("./cola.jl")
include("./sst.jl")
include("./mrpc.jl")
include("./qqp.jl")
include("./sts.jl")
include("./mnli.jl")
include("./snli.jl")
include("./qnli.jl")
include("./rte.jl")
include("./wnli.jl")
include("./diagnostic.jl")

end
