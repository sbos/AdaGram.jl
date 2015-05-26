function likelihood(vm::VectorModel, doc::DenseArray{Tw},
		window_length::Int, min_prob::Float64=1e-5)

	N = length(doc)
	if N == 1 return (0., 0) end

	z = zeros(T(vm))

	m = MeanCounter(Float64)

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
			add!(m, log(sum(local_ll)))
		end
	end
	return mean(m), m.n
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

function parallel_likelihood(vm::VectorModel, dict::Dictionary, path::String,
		window_length::Int, min_prob::Float64=1e-5; batch::Int=16777216, 
		log::Union(Nothing, String)=nothing)
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

	log_file = if log == nothing STDOUT else open(log, "w") end
	total_n = 0
	total_mean = Kahan(Float64)
	for (ll, n) in stats
		total_n += n
		println(log_file, ll)
		d = ll - sum(total_mean)
		add!(total_mean, n * d / total_n)
	end
	if log_file != STDOUT close(log_file) end

	return sum(total_mean)
end
