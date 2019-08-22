# Transformers.Datasets (not complete)
Functions for loading some common Datasets

## Provide datasets

- WMT
  - WMT14 (by Google Brain)
- IWSLT
  - IWSLT 2016
    - en <=> de
    - en <=> cs
    - en <=> fr
    - en <=> ar
- GLUE
  - CoLA
  - Diagnostic
  - GLUE
  - MNLI
  - MRPC
  - QNLI
  - QQP
  - RTE
  - SNLI
  - SST
  - STS
  - WNLI

## example

```julia
using Transformers.Datasets
using Transformers.Datasets.GLUE

task = GLUE.QNLI()
datas = dataset(Train, task)
get_batch(datas, 4)
```



## API reference

```@autodocs
Modules=[Transformers.Datasets]
Order = [:type, :function, :macro]
```
