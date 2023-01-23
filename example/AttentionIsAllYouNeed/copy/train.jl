# configuration
const N = 2
const V = 10
const Smooth = 1e-6
const Batch = 32
const lr = 1e-4

# text encoder / preprocess
const startsym = "11"
const endsym = "12"
const unksym = "0"
const labels = [unksym, startsym, endsym, map(string, 1:V)...]

const textenc = TransformerTextEncoder(split, labels; startsym, endsym, unksym, padsym = unksym)

function gen_data()
    global V
    d = join(rand(1:V, 10), ' ')
    (d,d)
end

# model definition
const hidden_dim = 512
const head_num = 8
const head_dim = 64
const ffn_dim = 2048

const token_embed = todevice(Embed(hidden_dim, length(textenc.vocab); scale = inv(sqrt(hidden_dim))))
const embed = Layers.CompositeEmbedding(token = token_embed, pos = SinCosPositionEmbed(hidden_dim))
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

const ps = params(trf_model)
const opt = ADAM(lr)

function train!()
    global Batch, trf_model
    println("start training")
    for i = 1:320*7
        data = batched([gen_data() for i = 1:Batch])
        input = preprocess(data)
        grad = gradient(ps) do
            nt = trf_model(input)
            shift_decode_loss(nt.logits, input.decoder_input.token, input.decoder_input.attention_mask)
        end
        i%8 == 0 && begin
            decode_loss = shift_decode_loss(trf_model(input).logits,
                                            input.decoder_input.token, input.decoder_input.attention_mask)
            @show decode_loss
        end
        update!(opt, ps, grad)
    end
end
