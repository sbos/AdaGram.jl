using AdaGram

vm, dict = load_model(ARGS[1])
window = int(ARGS[2])

z = zeros(T(vm))
while true
	sent = String[]
	labels = String[]
	while true
		line = strip(readline())
		if length(line) == 0 break end
		
		i = findfirst(line, ' ')
		push!(sent, line[1:i-1])
		push!(labels, line[i+1:end])
	end
	if length(sent) == 0 break end
	
	words = zeros(Int32, length(sent))
	for i in 1:length(sent)
		words[i] = get(dict.word2id, sent[i], 1) # *UNKNOWN*
	end

	for i in 1:length(sent)
		ctx = Int32[]
		for j in max(1, i - window):min(length(sent), i + window)
			push!(ctx, words[j])
		end
		z = disambiguate(vm, words[i], ctx)
		word = if words[i] == 1 "*UNKNOWN*" else string(dict.id2word[words[i]], "_", indmax(z)) end
		println(word, " ", labels[i])
	end
	println()
end
