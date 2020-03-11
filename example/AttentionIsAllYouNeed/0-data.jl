using ArgParse

using Random
Random.seed!(0)

using Transformers.Datasets
using Transformers.Datasets: WMT, IWSLT

function parse_commandline()
  s = ArgParseSettings()

  @add_arg_table s begin
    "--gpu", "-g"
    help = "use gpu"
    action = :store_true
    
    "task"
    help = "task name"
    required = true
    range_tester = x-> x ∈ ["wmt14", "iwslt2016", "copy"]
  end

  return parse_args(ARGS, s)
end

const args = parse_commandline()

enable_gpu(args["gpu"])

if args["task"] == "copy"
  const N = 2
  const V = 10
  const Smooth = 1e-6
  const Batch = 32
  const lr = 1e-4

  startsym = 11
  endsym = 12
  unksym = 0
  labels = [unksym, startsym, endsym, collect(1:V)...]

  function gen_data()
    global V
    d = rand(1:V, 10)
    (d,d)
  end

  function preprocess(data)
      x, t = data
      x = mkline.(x)
      t = mkline.(t)
      x_mask = getmask(x)
      t_mask = getmask(t)
      x, t = vocab(x, t)
     todevice(x,t,x_mask,t_mask)
  end

  function train!()
    global Batch
    println("start training")
    model = (embed=embed, encoder=encoder, decoder=decoder)
    i = 1
    for i = 1:320*7
      data = batched([gen_data() for i = 1:Batch])
      x, t, x_mask, t_mask = preprocess(data)
      grad = gradient(ps) do
          l = loss(model, x, t, x_mask, t_mask)
          l
      end
        i%8 == 0 && @show loss(model, x, t, x_mask, t_mask)
      update!(opt, ps, grad)
    end
  end

  mkline(x) = [startsym, x..., endsym]
elseif args["task"] == "wmt14" || args["task"] == "iwslt2016"
  const N = 6
  const Smooth = 0.4
  const Batch = 8
  const lr = 1e-6
  const MaxLen = 100

  const task = args["task"]

  if task == "wmt14"
    wmt14 = WMT.GoogleWMT()

    datas = dataset(Train, wmt14)
    vocab = get_vocab(wmt14)
  else
    iwslt2016 = IWSLT.IWSLT2016(:en, :de)

    datas = dataset(Train, iwslt2016)
    vocab = get_vocab(iwslt2016)
  end

  startsym = "<s>"
  endsym = "</s>"
  unksym = "</unk>"
  labels = [unksym, startsym, endsym, collect(keys(vocab))...]

  function preprocess(batch)
      x = mkline.(batch[1])
      t = mkline.(batch[2])
      x_mask = getmask(x)
      t_mask = getmask(t)
      x, t = vocab(x, t)
      todevice(x,t,x_mask,t_mask)
  end

  function train!()
    global Batch
    println("start training")
    i = 1
    model = (embed=embed, encoder=encoder, decoder=decoder)
    while (batch = get_batch(datas, Batch)) != []
      x, t, x_mask, t_mask = preprocess(batch)
      grad = gradient(ps) do
          loss(model, x, t, x_mask, t_mask)
      end
      i+=1
      i%8 == 0 && @show loss(model, x, t, x_mask, t_mask)

      @time update!(opt, ps, grad)
    end
  end

  if task == "wmt14"
    function mkline(x)
      global MaxLen
      xi = split(x)
      if length(xi) > MaxLen
        xi = xi[1:100]
      end

      [startsym, xi..., endsym]
    end
  else
    function mkline(x)
      global MaxLen
      xi = tokenize(x)
      if length(xi) > MaxLen
        xi = xi[1:100]
      end

      [startsym, xi..., endsym]
    end
  end
else
  error("task not define")
end
