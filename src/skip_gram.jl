function log_skip_gram(vm::VectorModel, w::T1, s::T2, v::T1) where
	{T1 <: Integer, T2 <: Integer}
	code = view(vm.code, :, v)
	path = view(vm.path, :, v)
	return ccall(_c_skip_gram, Float32,
		(Ptr{Float32}, Ptr{Float32},
			Int,
			Ptr{Int32}, Ptr{Int8}, Int),
		view(vm.In, :, s, w), vm.Out,
			M(vm),
			path, code, size(vm.code, 1))
end

export log_skip_gram
