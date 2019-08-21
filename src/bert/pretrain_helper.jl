using Random

using WordTokenizers


"""
    recursive_readdir(path::AbstractString="./")

recursive read all file from a dir. return a list of filenames.
"""
function recursive_readdir(path::AbstractString="./")
  ret = String[]
  for (root, dirs, files) in walkdir(path)
    append!(ret, map(file->joinpath(root, file), files))
  end
  ret
end

"""
    bert_pretrain_task(datachn::Channel, wordpiece::WordPiece;
                       buffer_size = 100,
                       channel_size = 100
                       wordpiece::WordPiece,
                       sentences_pool = sentences;
                       start_token = "[CLS]",
                       sep_token = "[SEP]",
                       mask_token = "[MASK]",
                       mask_ratio = 0.15,
                       real_token_ratio = 0.1,
                       random_token_ratio = 0.1,
                       whole_word_mask = false,
                       next_sentence_ratio = 0.5,
                       next_sentence = true,
                       tokenizer = tokenize,
                       istokenized = false,
                       return_real_sentence = false)

helper function to generate bert mask language modeling and next sentence prediction data. `datachn` is a `Channel` with input documents line by line.
"""
function bert_pretrain_task(datachn::Channel,
                       wordpiece::WordPiece;
                       buffer_size = 100,
                       channel_size = 100,
                       kwargs...
                       )
  outchn = Channel(channel_size)
  bert_pretrain_task(outchn, datachn, wordpiece; buffer_size = buffer_size, kwargs...)
  outchn
end

function bert_pretrain_task(outchn::Channel,
                       datachn::Channel,
                       wordpiece::WordPiece;
                       buffer_size = 100,
                       kwargs...
                       )
  task = @async begin
    buffer = Vector(undef, buffer_size)
    while isopen(datachn)
      i = 1
      eod = false
      while i <= buffer_size
        try
          sentence = take!(datachn)
          if isempty(sentence)
            continue
          else
            buffer[i] = sentence
            i+=1
          end
        catch e
          if isa(e, InvalidStateException) && e.state==:closed
            eod = true
            break
          else
            rethrow()
          end
        end
      end

      i -= 1

      if eod || i == buffer_size
        bert_pretrain_task(outchn, @view(buffer[1:(eod ? i - 1 : i)]), wordpiece; kwargs...)
      end
    end
  end
  bind(outchn, task)
end


function bert_pretrain_task(sentences,
                       wordpiece::WordPiece,
                       sentences_pool = sentences;
                       channel_size = 100,
                       kwargs...
                       )
  chn = Channel(channel_size)
  task = @async bert_pretrain_task(chn, sentences, wordpiece, sentences_pool; kwargs...)
  bind(chn, task)
  chn
end

function bert_pretrain_task(chn::Channel,
                       sentences,
                       wordpiece::WordPiece,
                       sentences_pool = sentences;
                       start_token = "[CLS]",
                       sep_token = "[SEP]",
                       mask_token = "[MASK]",
                       mask_ratio = 0.15,
                       real_token_ratio = 0.1,
                       random_token_ratio = 0.1,
                       whole_word_mask = false,
                       next_sentence_ratio = 0.5,
                       next_sentence = true,
                       tokenizer = tokenize,
                       istokenized = false,
                       return_real_sentence = false)

  foreach(enumerate(sentences)) do (i, sentence)
    sentenceA = masksentence(
      istokenized ? sentence : tokenizer(sentence),
      wordpiece;
      mask_token = mask_token,
      mask_ratio = mask_ratio,
      real_token_ratio = real_token_ratio,
      random_token_ratio = random_token_ratio,
      whole_word_mask = whole_word_mask
    )

    if next_sentence
      if rand() <= next_sentence_ratio && i != length(sentences)
        sentenceB = sentences[i+1]
        isnext = true
      else
        sentenceB = rand(sentences_pool)
        isnext = false
      end

      sentenceB = masksentence(
        istokenized ? sentenceB : tokenizer(sentenceB),
        wordpiece;
        mask_token = mask_token,
        mask_ratio = mask_ratio,
        real_token_ratio = real_token_ratio,
        random_token_ratio = random_token_ratio,
        whole_word_mask = whole_word_mask
      )

      masked_sentence = _wrap_sentence(sentenceA[1],
                                       sentenceB[1];
                                       start_token = start_token,
                                       sep_token = sep_token)

      sentence = _wrap_sentence(sentenceA[2],
                                sentenceB[2];
                                start_token = start_token,
                                sep_token = sep_token)

      mask_idx = _wrap_idx(sentenceA[3],
                           sentenceB[3],
                           length(sentenceA[1]))
    else
      masked_sentence = _wrap_sentence(sentenceA[1];
                                       start_token = start_token,
                                       sep_token = sep_token)

      sentence = _wrap_sentence(sentenceA[2];
                                start_token = start_token,
                                sep_token = sep_token)

      mask_idx = _wrap_idx(sentenceA[3])
    end

    masked_token = sentence[mask_idx]

    if return_real_sentence
      if next_sentence
        put!(chn, (masked_sentence, mask_idx, masked_token, isnext, sentence))
      else
        put!(chn, (masked_sentence, mask_idx, masked_token, sentence))
      end
    else
      if next_sentence
        put!(chn, (masked_sentence, mask_idx, masked_token, isnext))
      else
        put!(chn, (masked_sentence, mask_idx, masked_token))
      end
    end
  end
end

function _wrap_sentence(sentence1, sentence2...; start_token = "[CLS]", sep_token = "[SEP]")
  pushfirst!(sentence1, start_token)
  push!(sentence1, sep_token)
  map(s->push!(s, sep_token), sentence2)
  vcat(sentence1, sentence2...)
end

_wrap_idx(sentence1_idx, pre_len = 1) = sentence1_idx .+= pre_len
function _wrap_idx(sentence1_idx, sentence2_idx, len1)
  _wrap_idx(sentence1_idx)
  _wrap_idx(sentence2_idx, len1)
  vcat(sentence1_idx, sentence2_idx)
end



masksentence(sentence::String,
             tokenizer::Tf, wordpiece::WordPiece;
             mask_token = "[MASK]",
             mask_ratio = 0.15,
             real_token_ratio = 0.1,
             random_token_ratio = 0.1,
             whole_word_mask = false) where Tf = masksentence(
               tokenizer(sentence),
               wordpiece;
               mask_token = mask_token,
               mask_ratio = mask_ratio,
               real_token_ratio = real_token_ratio,
               random_token_ratio = random_token_ratio,
               whole_word_mask = whole_word_mask)

function masksentence(words,
                      wordpiece;
                      mask_token = "[MASK]",
                      mask_ratio = 0.15,
                      real_token_ratio = 0.1,
                      random_token_ratio = 0.1,
                      whole_word_mask = false)

  if whole_word_mask
    masked_word_idx = randsubseq(1:length(words), mask_ratio)

    is_masked = fill(false, length(words))
    map(masked_word_idx) do i
      is_masked[i] = true
    end

    tokens = Vector{String}()
    masked_idx = Vector{Int}()

    sizehint!(tokens, length(words))
    sizehint!(masked_idx, length(masked_word_idx))

    len = 0
    for (word, im) ∈ zip(words, is_masked)
      wps = wordpiece(word)
      wpl = length(wps)
      append!(tokens, wps)
      im && append!(masked_idx, map(i->i+len, 1:wpl))
      len += wpl
    end
  else
    tokens = wordpiece(words)
    masked_idx = randsubseq(1:length(tokens), mask_ratio)
  end

  masked_tokens = copy(tokens)

  for idx ∈ masked_idx
    r = rand()
    if r <= random_token_ratio
      masked_tokens[idx] = rand(wordpiece.vocab)
    elseif r > real_token_ratio + random_token_ratio
      masked_tokens[idx] = mask_token
    end
  end

  return masked_tokens, tokens, masked_idx
end

