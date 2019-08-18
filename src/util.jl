function read_from_file(vocab_path::AbstractString, min_freq::Int64=0,
	stopwords::Set{AbstractString}=Set{AbstractString}(); regex::Regex=r"")
	fin = open(vocab_path)
	freqs = Array{Int64, 1}()
	id2word = Array{AbstractString, 1}()
	while !eof(fin)
		try
			word, freq = split(readline(fin))
			freq_num = parse(Int64, freq)
			if freq_num < min_freq || word in stopwords ||
				match(regex, word) == nothing continue end
			push!(id2word, word)
			push!(freqs, freq_num)
		catch e
		end
	end
	close(fin)

	return freqs, id2word
end

function read_from_file(vocab_path::AbstractString, M::Int, T::Int,
	alpha::Float64, d::Float64=0., min_freq::Int=5,
	removeTopK::Int=70, stopwords::Set{AbstractString}=Set{AbstractString}();
	regex::Regex=r"")
	freqs, id2word = read_from_file(vocab_path, min_freq, stopwords; regex=regex)

	S = sortperm(freqs, rev=true)
	freqs = freqs[S[removeTopK+1:end]]
	id2word = id2word[S[removeTopK+1:end]]

	return VectorModel(freqs, M, T, alpha, d), Dictionary(id2word)
end

function build_from_file(text_path::AbstractString, M::Int, T::Int, min_freq::Int64=5)
	f = open(text_path)
	freqs, id2word = count_words(f)
	close(f)

	return VectorModel(freqs, M, T), Dictionary(id2word)
end

function dict_from_file(vocab_path::AbstractString)
	freqs, id2word = read_from_file(vocab_path)

	return Dictionary(id2word)
end

function read_word2vec(path::AbstractString)
	fin = open(path)

	line = readline(fin)
	line = split(line)

	V = parse(Int64, line[1])
	M = parse(Int64, line[2])

	In = zeros(Float32, M, V)
	id2word = Array{AbstractString, 1}()

	for v in 1:V
		word = readuntil(fin, ' ')[1:end-1]
		push!(id2word, word)

		In[:, v] = read(fin, Float32, (M))
		readuntil(fin, '\n')
	end

	close(fin)

	return In, Dictionary(id2word)
end

function write_word2vec(path::AbstractString, vm::VectorModel, dict::Dictionary)
	fout = open(path, "w")
	write(fout, "$(V(vm)) $(M(vm))\n")
	for v in 1:V(vm)
		write(fout, "$(dict.id2word[v]) ")
		for i in 1:M(vm)
			write(fout, vm.In[i, 1, v])
		end
		write(fout, "\n")
	end
	close(fout)
end

function finalize!(vm::VectorModel)
	vm.frequencies = sdata(vm.frequencies)
	vm.In = sdata(vm.In)
	vm.Out = sdata(vm.Out)
	vm.counts = sdata(vm.counts)
	vm.code = sdata(vm.code)
	vm.path = sdata(vm.path)
end

function save_model(path::AbstractString, vm::VectorModel, dict::Dictionary, min_prob=1e-5)
	file = open(path, "w")
	println(file, V(vm), " ", M(vm), " ", T(vm))
	println(file, vm.alpha, " ", vm.d)
	println(file, size(vm.code, 1))

	write(file, vm.frequencies)
	write(file, vm.code)
	write(file, vm.path)
	write(file, vm.counts)
	write(file, vm.Out)

	z = zeros(T(vm))

	for v in 1:V(vm)
		nsenses = expected_pi!(z, vm, v, min_prob)
		println(file, dict.id2word[v])
		println(file, nsenses)
		for k in 1:T(vm)
			if z[k] < min_prob continue end
			println(file, k)
			write(file, view(vm.In, :, k, v))
			println(file)
		end
	end

	close(file)
end

function load_model(path::AbstractString)
	file = open(path)

	_V, _M, _T = map(x -> parse(Int, x), split(readline(file)))
	alpha, d = map(x -> parse(Float64, x) , split(readline(file)))
	max_length = parse(Int, readline(file))

	vm = VectorModel(max_length, _V, _M, _T, alpha, d)
	read!(file, sdata(vm.frequencies))
	read!(file, sdata(vm.code))
	read!(file, sdata(vm.path))
	read!(file, sdata(vm.counts))
	read!(file, sdata(vm.Out))

	buffer = zeros(Float32, M(vm))

	id2word = Array{AbstractString, 1}()
	for v in 1:V(vm)
		word = strip(readline(file))
		nsenses = parse(Int, readline(file))
		push!(id2word, word)

		for r in 1:nsenses
			k = parse(Int, readline(file))
			read!(file, buffer)
			vm.In[:, k, v] = buffer
			readline(file)
		end
	end

	close(file)

	return vm, Dictionary(id2word)
end

function preprocess(vm::VectorModel, doc::Array{Int32}; min_freq::Int64=5,
	subsampling_treshold::Float64 = 1e-5)
	data = Array(Int32, 0)
	total_freq = sum(vm.frequencies)

	for i in 1:length(doc)
		assert(1 <= doc[i] <= length(vm.frequencies))
		if vm.frequencies[doc[i]] < min_freq
			continue
		end

		if rand() < 1. - sqrt(subsampling_treshold / (vm.frequencies[doc[i]] / total_freq))
			continue
		end

		push!(data, doc[i])
	end

	return data
end

function vec(vm::VectorModel, v::Integer, s::Integer)
	x = vm.In[:, s, v]
	return x / norm(x)
end

function vec(vm::VectorModel, dict::Dictionary, w::AbstractString, s::Integer)
	return vec(vm, dict.word2id[w], s)
end

function nearest_neighbors(vm::VectorModel, dict::Dictionary, word::DenseArray{Tsf},
		K::Integer=10; exclude::Array{Tuple{Int32, Int64}}=Array(Tuple{Int32, Int64}, 0),
		min_count::Float64=1.)
	sim = zeros(Tsf, (T(vm), V(vm)))

	for v in 1:V(vm)
		for s in 1:T(vm)
			if vm.counts[s, v] < min_count
				sim[s, v] = -Inf
				continue
			end
			in_vs = view(vm.In, :, s, v)
			sim[s, v] = dot(in_vs, word) / norm(in_vs)
			@assert(!isnan(sim[s, v]), "NaN found, $s, $(dict.id2word[v])")
		end
	end
	for (v, s) in exclude
		sim[s, v] = -Inf
	end
	top = Array{Tuple{Int, Int}}(undef, K)
	topSim = zeros(Tsf, K)

	for k in 1:K
		curr_max = argmax(sim)
		topSim[k] = sim[curr_max[1], curr_max[2]]
		sim[curr_max[1], curr_max[2]] = -Inf

		top[k] = curr_max
	end
	return Tuple{AbstractString, Int, Tsf}[(dict.id2word[r[2]], r[1], simr)
		for (r, simr) in zip(top, topSim)]
end

function nearest_neighbors(vm::VectorModel, dict::Dictionary,
		w::AbstractString, s::Int, K::Integer=10; min_count::Float64=1.)
	v = dict.word2id[w]
	return nearest_neighbors(vm, dict, vec(vm, v, s), K;
		exclude=[(v, s)], min_count=min_count)
end

cos_dist(x, y) = 1. - dot(x, y) / norm(x, 2) / norm(y, 2)

function disambiguate(vm::VectorModel, x::Tw,
		context::AbstractArray{Tw, 1}, use_prior::Bool=true,
		min_prob::Float64=1e-3) where {Tw <: Integer}
	z = zeros(T(vm))

	if use_prior
		expected_pi!(z, vm, x)
		for k in 1:T(vm)
			if z[k] < min_prob
				z[k] = 0.
			end
			z[k] = log(z[k])
		end
	end
	for y in context
		var_update_z!(vm, x, y, z)
	end

	exp_normalize!(z)

	return z
end

function disambiguate(vm::VectorModel, dict::Dictionary, x::AbstractString,
	context::AbstractArray{Ts, 1}, use_prior::Bool=true,
	min_prob::Float64=1e-3) where {Ts <: AbstractString}
	return disambiguate(vm, dict.word2id[x], Int32[dict.word2id[y] for y in context], use_prior, min_prob)
end

export nearest_neighbors
export disambiguate
export pi, write_extended
export cos_dist, preprocess, read_word2vec, write_word2vec
