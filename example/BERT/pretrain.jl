using Transformers
using Transformers.Basic
using Transformers.Pretrain
using Transformers.Datasets
using Transformers.BidirectionalEncoder

using Flux
using Flux: onehotbatch, gradient
import Flux.Optimise: update!
using WordTokenizers
using CuArrays

include("./model.jl")

const Epoch = 20
const Batch = 4
const FromScratch = false

#use wordpiece and tokenizer from pretrain
const wordpiece = pretrain"bert-uncased_L-12_H-768_A-12:wordpiece"
const tokenizer = pretrain"bert-uncased_L-12_H-768_A-12:tokenizer"
const vocab = Vocabulary(wordpiece)

#see model.jl
const bert_model = gpu(
  FromScratch ? create_bert() : pretrain"bert-uncased_L-12_H-768_A-12:bert_model"
)
const ps = params(bert_model)
const opt = ADAM(1e-4)

#read sentence line by line and wrap in channel
#for working with files, read the data into a channel
function prepare_data()
  Datasets.reader("./sample_text.txt")
end

function preprocess(batch)
  mask = getmask(batch[1])
  tok = vocab(batch[1])
  segment = fill!(similar(tok), 1.0)

  for (i, sentence) ∈ enumerate(batch[1])
    j = findfirst(isequal("[SEP]"), sentence)
    if j !== nothing
      @view(segment[j+1:end, i]) .= 2.0
    end
  end

  ind = vcat(
    map(enumerate(batch[2])) do (i, x)
     map(j->(j,i), x)
    end...)

  masklabel = onehotbatch(vocab(vcat(batch[3]...)), 1:length(vocab))
  nextlabel = onehotbatch(batch[4], (true, false))

  return (tok=tok, segment=segment), ind, masklabel, nextlabel, mask
end

function loss(data, ind, masklabel, nextlabel, mask = nothing)
  e = bert_model.embed(data)
  t = bert_model.transformers(e, mask)
  nextloss = Basic.logcrossentropy(
    bert_model.classifier.nextsentence(
      bert_model.classifier.pooler(
        t[:,1,:]
      )
    ),
    nextlabel
  )
  mkloss = masklmloss(bert_model.embed.embeddings.tok,
                      bert_model.classifier.masklm.transform,
                      bert_model.classifier.masklm.output_bias,
                      t,
                      ind,
                      masklabel
                      )
  return nextloss + mkloss
end

function train!()
  global Batch
  global Epoch
  @info "start training"
  for e = 1:Epoch
    @info "epoch: $e"
    #prepare training data
    pretrain_data = prepare_data()

    #pretrain_task function help you to create mask lm and next sentence prediction data
    datas = bert_pretrain_task(pretrain_data, wordpiece; tokenizer = tokenizer) #or set_tokenizer(tokenizer)

    i = 1
    while(batch = get_batch(datas, Batch)) !== nothing
      data, ind, masklabel, nextlabel, mask = todevice(preprocess(batch))
      l = loss(data, ind, masklabel, nextlabel, mask)
      @show l
      grad = gradient(()->l, ps)
      update!(opt, ps, grad)
    end
  end
end

train!()
