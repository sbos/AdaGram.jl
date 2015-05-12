mean_beta(a, b) = a / (a + b)
meanlog_beta(a, b) = digamma(a) - digamma(a + b)
mean_mirror(a, b) = mean_beta(b, a)
meanlog_mirror(a, b) = meanlog_beta(b, a)

function expected_logpi!{Tw <: Integer}(pi::Vector{Float64}, vm::VectorModel, w::Tw, min_prob::Float64=1e-3)
	r = 0.
	x = 1.
	senses = 0
	pi[:] = vm.counts[:, w] #view(vm.counts, :, w)
	ts = sum(pi)
	for k in 1:T(vm)-1
		ts = max(ts - pi[k], 0.)
		a, b = 1. + pi[k] - vm.d, vm.alpha + k*vm.d + ts
		pi[k] = meanlog_beta(a, b) + r
		r += meanlog_mirror(a, b)

		pi_k = mean_beta(a, b) * x
		x = max(x - pi_k, 0.)
		if pi_k >= min_prob
			senses += 1
		end
	end
	pi[T(vm)] = r
	if x >= min_prob senses += 1 end
	return senses
end 

function expected_pi!{Tw <: Integer}(pi::Vector{Float64}, vm::VectorModel, 
		w::Tw, min_prob=1e-3)
	r = 1.
	senses = 0
	ts = sum(view(vm.counts, :, w))
	for k in 1:T(vm)-1
		ts = max(ts - vm.counts[k, w], 0.)
		a, b = 1. + vm.counts[k, w] - vm.d, vm.alpha + k*vm.d + ts
		pi[k] = mean_beta(a, b) * r
		if pi[k] >= min_prob senses += 1 end
		r = max(r - pi[k], 0.)
	end
	pi[T(vm)] = r
	if r >= min_prob senses += 1 end
	return senses
end

function expected_pi{Tw <: Integer}(vm::VectorModel, w::Tw, min_prob=1e-3)
	z = zeros(T(vm))
	expected_pi!(z, vm, w, min_prob)
	return z
end
