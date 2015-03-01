function word_iterator(f::IO, end_pos::Int64=-1)
  function producer()
    while (end_pos < 0 || position(f) < end_pos) && !eof(f)
      w = readuntil(f, ' ')
      if length(w) < 1 break end
      w = w[1:end-1]
      if !isblank(w)
        produce(w)
      end
    end
  end

  return Task(producer)
end

function looped_word_iterator(f::IO, start_pos::Int64, end_pos::Int64)
  function producer()
    while true
      w = readuntil(f, ' ')
      if length(w) < 1 break end
      w = w[1:end-1]
      if !isblank(w)
        produce(w)
      end
      if position(f) >= end_pos seek(f, start_pos) end
    end
  end

  return Task(producer)
end

function count_words(f::IOStream, min_freq::Int=5)
  counts = Dict{String, Int64}()

  for word in word_iterator(f)
    if get(counts, word, 0) == 0
      counts[word] = 1
    else
      counts[word] += 1
    end
  end

  for word in [keys(counts)...]
    if counts[word] < min_freq
      delete!(counts, word)
    end
  end

  V = length(counts)
  id2word = Array(String, V)
  freqs = zeros(Int64, V)
  i = 1
  for (word, count) in counts
    id2word[i] = word
    freqs[i] = count
    i += 1
  end

  return freqs, id2word
end

function align(f::IO)
  while !isblank(read(f, Char))
    continue
  end

  while isblank(read(f, Char))
    continue
  end

  seek(f, position(f)-1)
end

function read_words(f::IO,
    dict::Dictionary, doc::DenseArray{Int32},
    batch::Int, last_pos::Int)
  words = word_iterator(f, last_pos)
  i = 1
  for j in 1:batch
    word = consume(words)
    id = get(dict.word2id, word, -1)
    if id == -1
      continue
    end

    doc[i] = id
    i += 1
  end

  return view(doc, 1:i-1)
end

function read_words(str::String,
    dict::Dictionary, doc::DenseArray{Int32},
    batch::Int, last_pos::Int)
  i = 1
  for word in split(str, ' ')
    id = get(dict.word2id, word, -1)
    if id == -1
      continue
    end

    doc[i] = id
    i += 1
  end

  return view(doc, 1:i-1)
end
function read_words(f::IOStream, start_pos::Int64, end_pos::Int64,
    dict::Dictionary, doc::DenseArray{Int32},
    freqs::DenseArray{Int64}, threshold::Float64,
    words_read::DenseArray{Int64}, total_words::Float64)
  words = looped_word_iterator(f, start_pos, end_pos)
  i = 1
  while i <= length(doc) && words_read[1] < total_words
    word = consume(words)
    id = get(dict.word2id, word, -1)
    if id == -1
      continue
    elseif rand() < 1. - sqrt(threshold / (freqs[id] / total_words))
      words_read[1] += 1
      continue
    end

    doc[i] = id
    i += 1
  end

  return view(doc, 1:i-1)
end
