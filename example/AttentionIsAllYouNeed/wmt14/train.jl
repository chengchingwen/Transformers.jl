using Transformers.Datasets: WMT

# configuration
const N = 1
const Smooth = 0.1
const Epoch = 1
const Batch = 32
const lr = 1e-5
const MaxLen = 100
const wmt14 = WMT.GoogleWMT()

# text encoder / preprocess
const word_counts = get_vocab(wmt14)
const startsym = "<s>"
const endsym = "</s>"
const unksym = "</unk>"
const labels = [unksym; startsym; endsym; collect(keys(word_counts))]

const textenc = TransformerTextEncoder(split, labels; startsym, endsym, unksym,
                                       padsym = unksym, trunc = MaxLen)

# model definition
const hidden_dim = 512
const head_num = 8
const head_dim = 64
const ffn_dim = 2048

const token_embed = todevice(Embed(hidden_dim, length(textenc.vocab); scale=inv(sqrt(hidden_dim))))
const pos_embed = todevice(FixedLenPositionEmbed(hidden_dim, MaxLen))
const embed = Layers.CompositeEmbedding(token = token_embed, pos = pos_embed)
const embed_decode = EmbedDecoder(token_embed)
const encoder = todevice(Transformer(TransformerBlock       , N, head_num, hidden_dim, head_dim, ffn_dim))
const decoder = todevice(Transformer(TransformerDecoderBlock, N, head_num, hidden_dim, head_dim, ffn_dim))

const seq2seq = Seq2Seq(encoder, decoder)
const trf_model = Layers.Chain(
    Layers.Parallel{(:encoder_input, :decoder_input)}(
        Layers.Chain(embed, todevice(Dropout(0.1)))),
    seq2seq,
    Layers.Branch{(:logits,)}(embed_decode),
)

const opt_rule = Optimisers.Adam(lr)
const opt = Optimisers.setup(opt_rule, trf_model)

function train!()
    global Epoch, Batch, trf_model
    println("start training")
    i = 1
    for e = 1:Epoch
        datas = dataset(Train, wmt14)
        while (batch = get_batch(datas, Batch)) |> !isnothing
            input = preprocess(batch)
            decode_loss, (grad,) = Zygote.withgradient(trf_model) do model
                nt = model(input)
                shift_decode_loss(nt.logits, input.decoder_input.token, input.decoder_input.attention_mask)
            end
            i += 1
            i % 8 == 0 && @show decode_loss
            Optimisers.update!(opt, trf_model, grad)
        end
    end
end
