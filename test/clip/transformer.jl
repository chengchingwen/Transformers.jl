@testset "CLIPTextTransformer" begin
    # Get a config from HGF
    clip_model_name = "openai/clip-vit-large-patch14"
    clip_config = Transformers.load_config(clip_model_name)
    clip_text_config = clip_config.text_config
    
    # Load CLIPTextEmbeddings
    embeddings = Transformers.HuggingFace.HGFCLIPTextEmbeddings(clip_config.text_config)
    state = Transformers.load_state(clip_model_name)
    HuggingFace.load_state!(embeddings, state.text_model.embeddings)
    
    # Tokenizer
    tkr = hgf"openai/clip-vit-large-patch14:tokenizer"
    # all token vectors returns a 77 element vector with padding
    tkr = Transformers.Basic.TransformerTextEncoder(tkr) do enc
        Pipelines(enc.process[1:5]) |> 
        Pipeline{:trunc_tok}(TextEncodeBase.trunc_or_pad(enc.trunc, enc.padsym), :tok) |>
        Pipelines(enc.process[7:end])
    end
    
    e = encode(tkr, [ "A photo of an astronaut riding a horse on mars", 
                      "A fantasy landscape, trending on artstation", 
                      "Face of a yellow cat, high resolution, sitting on a park bench"])
    s = reinterpret(Int32, e.input.tok)
    emb_123 = embeddings(s)
    emb_1 = embeddings(s[:, 1])
    
    @test size(emb_123) == (768, 77, 3)
    @test size(emb_1) == (768, 77)
    @test emb_123[:, :, 1] ≈ emb_1
    
    # Create all models
    clip_mlp = HuggingFace.HGFCLIPMLP(clip_config.text_config)
    clip_attn = HuggingFace.HGFCLIPAttention(clip_config.text_config)
    encoder_layer = HuggingFace.HGFCLIPEncoderLayer(clip_config.text_config)
    encoder = HuggingFace.HGFCLIPEncoder(clip_config.text_config)
    text_transformer = HuggingFace.HGFCLIPTextTransformer(clip_config.text_config)
    text_model = HuggingFace.HGFCLIPTextModel(clip_config.text_config)
    
    # Load all models
    Transformers.HuggingFace.load_state!(clip_mlp, state.text_model.encoder.layers[1].mlp)
    Transformers.HuggingFace.load_state!(clip_attn, state.text_model.encoder.layers[1].self_attn)
    Transformers.HuggingFace.load_state!(encoder_layer, state.text_model.encoder.layers[1])
    Transformers.HuggingFace.load_state!(encoder, state.text_model.encoder)
    Transformers.HuggingFace.load_state!(text_transformer, state.text_model)
    Transformers.HuggingFace.load_state!(text_model, state)
    
    # Forward calls
    # BS=1 and BS=3 should give same results
    result123 = text_model(s)     # BS=3 
    result1 = text_model(s[:, 1]) # BS=1
    @test result123.last_hidden_state[:, :, 1] ≈ result1.last_hidden_state
    
    # CLIPTextTransformer and CLIPTextModel should give same results
    result = text_transformer(s)
    @test result123.last_hidden_state ≈ result.last_hidden_state
    
    # Get intermediate results from CLIPTextModel
    result_with_hs = text_model(s; output_hidden_states=true)
    
    # First value in the intermediate results is the embedding
    @test result_with_hs.hidden_states[1] ≈ emb_123
    
    # Second value in the intermediate results is the first encoder layer's output with causal mask
    result_enc_layer = encoder_layer(emb_123; causal_attention_mask=NeuralAttentionlib.CausalMask())
    @test result_enc_layer[1] ≈ result_with_hs.hidden_states[2]
    
    # Final value in the intermediate results is the Encoder's last_hidden_state
    result_enc = encoder(emb_123; causal_attention_mask=NeuralAttentionlib.CausalMask())
    @test result_with_hs.hidden_states[13] ≈ result_enc.last_hidden_state
    
    result_attn = clip_attn(randn(768, 13, 3))
    @test size(result_attn[1]) == (768, 13, 3)
    result_attn = clip_attn(randn(768, 13, 3); output_attentions=true)
    @enter clip_attn(randn(768, 13, 3); output_attentions=true)
    @test size(result_attn[2]) == (13, 13, 12, 3) # seq_len=13, heads=12, batch_size=3 (seq_len, seq_len, heads, batch_size)
    result_mlp = clip_mlp(randn(768, 13, 3))
    @test size(result_mlp) == (768, 13, 3)
end
