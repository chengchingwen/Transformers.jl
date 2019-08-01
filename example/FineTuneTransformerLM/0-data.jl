using ArgParse

using Transformers.Datasets
using Transformers.Datasets: StoryCloze

const Batch = 4

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--gpu", "-g"
            help = "use gpu"
            action = :store_true
        "--epoch", "-e"
            help = "epoch"
            arg_type = Int
            default = 3
        "task"
            help = "task name"
            required = true
            range_tester = x -> x ∈ ["rocstories"]
    end

    return parse_args(ARGS, s)
end

const args = parse_commandline()

if args["gpu"]
    @eval using CuArrays
end

function transform(s1, s2, s3, s4, c1, c2, y)
    x = [startsym;
         segment(bpe, s1);
         segment(bpe, s2);
         segment(bpe, s3);
         segment(bpe, s4);
         delisym]
    x1 = [x; segment(bpe, c1); clfsym]
    x2 = [x; segment(bpe, c2); clfsym]

    x1, x2, y
end

function preprocess(batch)
    tdb = transform.(batch...)
    b1, b2, y = batched(tdb)
    b1_mask = getmask(b1)
    b2_mask = getmask(b2)
    c1i = [(findfirst(isequal(clfsym), x), i) for (i, x) in enumerate(b1)]
    c2i = [(findfirst(isequal(clfsym), x), i) for (i, x) in enumerate(b2)]
    b1, b2 = vocab(b1,b2)
    y = onehotbatch(y, labels)

    return b1, b2, c1i, c2i, y, b1_mask, b2_mask
end

function test()
    Flux.testmode!(gpt)
    println("eval:")
    i::Int = 0
    al::Float64 = 0.
    devl = dataset(Test, rocs)
    while (batch = get_batch(devl, Batch)) !== nothing
        b1, b2, c1i, c2i, y, b1_mask, b2_mask, = todevice(preprocess(batch))

        _, p = loss(b1, b2, y, b1_mask, b2_mask, c1i, c2i)
        a = acc(p, y)
        al += a
        i += 1
    end
    al /= i
    Flux.testmode!(gpt, false)
    @show al
end

function train!(epoch)
    global Batch, rocs, opt, ps
    for e = 1:epoch
        println("start training: $e")
        datas = dataset(Train, rocs)
        i::Int = 0
        al::Float64 = 0.
        while (batch = get_batch(datas, Batch)) !== nothing
            b1, b2, c1i, c2i, y, b1_mask, b2_mask, = todevice(preprocess(batch))

            l, p = loss(b1, b2, y, b1_mask, b2_mask, c1i, c2i)
            a = acc(p, y)
            al += a
            grad = gradient(()->l, ps)
            i+=1
            update!(opt, ps, grad)
            i%16==0 && @show al/i
        end
        test()
    end
end
