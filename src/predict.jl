function likelihood(vm::VectorModel, doc::DenseArray{Tw},
		window_length::Int, min_prob::Float64=1e-5)

	N = length(doc)
	if N == 1 return (0., 0) end

	z = zeros(T(vm))
	ll = Kahan(Float64)

	#counting number of word predictions in the batch
	n = 0
	if N >= 2 * window_length
		n = window_length * (N - 2*window_length) + window_length * (3 * window_length - 1)
	else
		for i in 1:N
			for j in max(1, i - window):min(N, i + window)
				if i == j continue end
				n += 1
			end
		end
	end

	for i in 1:N
		x = doc[i]

		window = window_length
		z[:] = 0.

		expected_pi!(z, vm, x)

		for j in max(1, i - window):min(N, i + window)
			if i == j continue end
			y = doc[j]

			local_ll = Kahan(Float64)
			for s in 1:T(vm)
				if z[s] < min_prob continue end
				In = view(vm, x, s)

				add!(local_ll, z[s] * exp(float64(log_skip_gram(vm, x, s, y))))
			end
			add!(ll, 1. / n * (log(sum(local_ll)) - sum(ll)))
		end
	end
	return sum(ll), n
end

function likelihood(vm::VectorModel, dict::Dictionary, f::IO,
		window_length::Int; batch::Int=16777216)
	buffer = zeros(Int32, batch)
	j = 0
	ll = 0.
	while !eof(f)
		doc = read_words(f, dict, buffer, length(buffer), -1)
		if length(doc) == 0 break end
		#println(j)
		local_ll, n = likelihood(vm, doc, window_length)
		ll += local_ll
		j += n
	end
	return ll / j
end

function text_likelihood(vm::VectorModel, dict::Dictionary, s::String,
		window_length::Int, batch::Int=16777216)
	buffer = zeros(Int32, batch)
	doc = read_words(s, dict, buffer, length(buffer), -1)
	ll, j = likelihood(vm, doc, window_length)
	return ll / j
end

export text_likelihood

function likelihood(vm::VectorModel, dict::Dictionary, path::String,
		window_length::Int; batch::Int=16777216)
	f = open(path)
	ll = likelihood(vm, dict, f, window_length; batch = batch)
	close(f)
	return ll
end

function parallel_likelihood(vm::VectorModel, dict::Dictionary, path::String,
		window_length::Int, min_prob::Float64=1e-5; batch::Int=16777216)
	nbytes = filesize(path)

	words_read = shared_zeros(Int64, (1,))
	stats = Array((Float64, Int64), 0)

	function do_work(id::Int)
		file = open(path)

		bytes_per_worker = convert(Int, floor(nbytes / nworkers()))

		start_pos = bytes_per_worker * (id-1)
		end_pos = start_pos+bytes_per_worker

		seek(file, start_pos)
		align(file)
		buffer = zeros(Int32, batch)
		local_stats = Array((Float64, Int64), 0)
		while true
			doc = read_words(file, dict, buffer, batch, end_pos)

			println("$(length(doc)) words read, $(position(file))/$end_pos")
			if length(doc) == 0 break end

			push!(local_stats, likelihood(vm, doc, window_length, min_prob))
		end

		close(file)
		return local_stats
	end

	refs = Array(RemoteRef, nworkers())
	for i in 1:nworkers()
		refs[i] = remotecall(i+1, do_work, i)
	end

	for i in 1:nworkers()
		append!(stats, fetch(refs[i]))
	end

	total_n = 0
	total_mean = Kahan(Float64)
	for (ll, n) in stats
		total_n += n
		d = ll - sum(total_mean)
		add!(total_mean, n * d / total_n)
	end

	return sum(total_mean)
end
