import LinearAlgebra.axpy!

function inplace_train_vectors!(vm::VectorModel, doc::AbstractArray{Tw},
		window_length::Int,
		start_lr::Float64, total_words::Float64, words_read::DenseArray{Int64},
		total_ll::DenseArray{Float64}; batch::Int=10000,
		context_cut::Bool = true, sense_treshold::Float64=1e-32)

	N = length(doc)
	in_grad = zeros(Tsf, M(vm), T(vm))
	out_grad = zeros(Tsf, M(vm))

	z = zeros(T(vm))
	senses = 0.
	max_senses = 0.

	start_time = time()
	for i in 1:N
		x = doc[i]
		lr1 = max(start_lr * (1 - words_read[1] / (total_words+1)), start_lr * 1e-4)
		lr2 = lr1

		random_reduce = context_cut ? rand(1:window_length-1) : 0
		window = window_length - random_reduce

		z[:] .= 0.

		n_senses = var_init_z!(vm, x, z)
		senses += n_senses
		max_senses = max(max_senses, n_senses)
		for j in max(1, i - window):min(N, i + window) #
			if i == j continue end
			var_update_z!(vm, x, doc[j], z)
		end

		exp_normalize!(z)

		for j in max(1, i - window):min(N, i + window)
			if i == j continue end
			y = doc[j]

			ll = in_place_update!(vm, x, y, z, lr1, in_grad, out_grad, sense_treshold)

			total_ll[2] += 1
			total_ll[1] += (ll - total_ll[1]) / total_ll[2]
		end

		words_read[1] += 1

		#variational update for q(pi_v)
		var_update_counts!(vm, x, z, lr2)

		if i % batch == 0
			time_per_kword = batch / (time() - start_time) / 1000
			@printf("%.2f%% %.4f %.4f %.4f %.2f/%.2f %.2f kwords/sec\n",
					words_read[1] / (total_words / 100),
					total_ll[1], lr1, lr2, senses / i, max_senses, time_per_kword)
			start_time = time()
		end

		if words_read[1] > total_words break end
	end
end

function in_place_update!(vm::VectorModel, x::Tw, y::Tw, z::DenseArray{Float64},
	lr::Float64, in_grad::DenseArray{Tsf, 2}, out_grad::DenseArray{Tsf},
	sense_treshold::Float64) where {Tw <: Integer}

	return ccall(_c_inplace_update, Float32,
		(Ptr{Float32}, Ptr{Float32},
			Int, Int, Ptr{Float64},
			Int,
			Ptr{Int32}, Ptr{Int8}, Int64,
			Ptr{Float32}, Ptr{Float32},
			Float32, Float32),
		sdata(vm.In), sdata(vm.Out),
			M(vm), T(vm), z,
			x,
			view(vm.path, :, y), view(vm.code, :, y), size(vm.code, 1),
			in_grad, out_grad,
			lr, sense_treshold)
end

function var_init_z!(vm::VectorModel, x::Integer, z::DenseArray{Float64})
	return expected_logpi!(z, vm, x)
end

function var_update_z!(vm::VectorModel, x::Tw, y::Tw, z::DenseArray{Float64},
	num_meanings::Int=T(vm)) where {Tw <: Integer}
 	ccall(_c_update_z, Cvoid,
		(Ptr{Float32}, Ptr{Float32},
			Int, Int, Ptr{Float64}, Int,
			Ptr{Int32}, Ptr{Int8}, Int64),
		vm.In, vm.Out,
			M(vm), num_meanings, z, x,
			view(vm.path, :, y), view(vm.code, :, y), size(vm.path, 1))
end

function var_update_counts!(vm::VectorModel, x::Integer,
		local_counts::DenseArray{Float64}, lr::Float64)
	counts = view(vm.counts, :, x)
	for k in 1:T(vm)
		counts[k] += lr * (local_counts[k] * vm.frequencies[x] - counts[k])
	end
end

function inplace_train_vectors!(vm::VectorModel, dict::Dictionary, path::AbstractString,
		window_length::Int; batch::Int = 64000, start_lr::Float64 = 0.025,
		log_path::Union{AbstractString, Nothing} = nothing,
		threshold::Float64 = Inf, context_cut::Bool = true, epochs::Int = 1,
		init_count::Float64=-1, sense_treshold::Float64=1e-32)
	for w in 1:V(vm)
		vm.counts[1, w] = init_count > 0 ? init_count : vm.frequencies[w]
	end

	nbytes = filesize(path)
	train_words = Float64(sum(vm.frequencies)) * epochs

	words_read = shared_zeros(Int64, (1,))
	total_ll = shared_zeros(Float64, (2,))

	function do_work(id::Int)
		file = open(path)

		bytes_per_worker = convert(Int, floor(nbytes / nworkers()))

		start_pos = bytes_per_worker * (id-1)
		end_pos = start_pos+bytes_per_worker

		seek(file, start_pos)
		align(file)
		buffer = zeros(Int32, batch)
		while words_read[1] < train_words
			doc = read_words(file, start_pos, end_pos, dict, buffer,
				vm.frequencies, threshold, words_read, train_words)

			println("$(length(doc)) words read, $(position(file))/$end_pos")
			if length(doc) == 0
				break
			end

			inplace_train_vectors!(vm, doc, window_length,
				start_lr, train_words, words_read, total_ll;
				context_cut = context_cut, sense_treshold = sense_treshold)
		end

		close(file)
	end

	refs = Array{Future, 1}(undef, nworkers())
	for i in 1:nworkers()
		refs[i] = remotecall(do_work, i+1, i)
	end

	for i in 1:nworkers()
		fetch(refs[i])
	end

	println("Learning complete $(words_read[1]) / $train_words")

	return words_read[1]
end
