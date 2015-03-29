module AdaGram

using Devectorize
using ArrayViews
using NumericExtensions

sigmoid(x) = 1. / (1. + exp(-x))
log_sigmoid(x) = -log(1. + exp(-x))

Tsf = Float32
Tw = Int32

include("softmax.jl")

import ArrayViews.view
import ArrayViews.Subs
import Base.vec

test(node::HierarchicalSoftmaxNode) = node.parent

view(x::SharedArray, i1::Subs, i2::Subs) = view(sdata(x), i1, i2)
view(x::SharedArray, i1::Subs, i2::Subs, i3::Subs) = view(sdata(x), i1, i2, i3)

type Dictionary
	word2id::Dict{String, Tw}
	id2word::Array{String}

	function Dictionary(id2word::Array{String})
		word2id = Dict{String, Int}()
		for v in 1:length(id2word)
			push!(word2id, id2word[v], v)
		end
		return new(word2id, id2word)
	end
end

type VectorModel
	frequencies::DenseArray{Int64}
	code::DenseArray{Int8, 2}
	path::DenseArray{Int32, 2}
	In::DenseArray{Tsf, 3}
	Out::DenseArray{Tsf, 2}
	alpha::Float64
	d::Float64
	counts::DenseArray{Float32, 2}
end

M(vm::VectorModel) = size(vm.In, 1) #dimensionality of word vectors
T(vm::VectorModel) = size(vm.In, 2) #number of meanings
V(vm::VectorModel) = size(vm.In, 3) #number of words

function shared_rand{T}(dims::Tuple, norm::T)
	S = SharedArray(T, dims; init = S -> begin
			chunk = localindexes(S)
			chunk_size = length(chunk)
			data = rand(chunk_size)
			subtract!(data, 0.5)
			divide!(data, norm)
			S[chunk] = data
		end)
	return S
end

function shared_zeros{T}(::Type{T}, dims::Tuple)
	S = SharedArray(T, dims; init = S -> begin
			chunk = localindexes(S)
			chunk_size = length(chunk)
			S[chunk] = 0.
		end)
	return S
end

function VectorModel(max_length::Int64, V::Int64, M::Int64, T::Int64=1, alpha::Float64=1e-2, 
		d::Float64=0.)
	path = shared_zeros(Int32, (max_length, V))
	code = shared_zeros(Int8, (max_length, V))

	code[:] = -1

	In = shared_rand((M, T, V), float32(M))
	Out = shared_rand((M, V), float32(M))

	counts = shared_zeros(Float32, (T, V))

	frequencies = shared_zeros(Int64, (V,))

	return VectorModel(frequencies, code, path, In, Out, alpha, d, counts)
end

function VectorModel(freqs::Array{Int64}, M::Int64, T::Int64=1, alpha::Float64=1e-2, 
	d::Float64=0.)
	V = length(freqs)

	nodes = build_huffman_tree(freqs)
	outputs = convert_huffman_tree(nodes, V)

	max_length = maximum(map(x -> length(x.code), outputs))

	path = shared_zeros(Int32, (max_length, V))
	code = shared_zeros(Int8, (max_length, V))

	for v in 1:V
		code[:, v] = -1
		for i in 1:length(outputs[v])
			code[i, v] = outputs[v].code[i]
			path[i, v] = outputs[v].path[i]
		end
	end

	In = shared_rand((M, T, V), float32(M))
	Out = shared_rand((M, V), float32(M))

	counts = shared_zeros(Float32, (T, V))

	frequencies = shared_zeros(Int64, (V,))
	frequencies[:] = freqs

	return VectorModel(frequencies, code, path, In, Out, alpha, d, counts)
end

view(vm::VectorModel, v::Integer, s::Integer) = view(vm.In, :, s, v)

include("skip_gram.jl")
include("stick_breaking.jl")
include("textutil.jl")
include("gradient.jl")
include("predict.jl")
include("util.jl")

export VectorModel, gradient!
export get_gradient, apply_gradient!
export V, T, M, L
export train_vectors!, inplace_train_vectors!
export vec, closest_words
export finalize!
export save_model, read_from_file, dict_from_file, build_from_file
export disambiguate, write_dictionary
export likelihood, parallel_likelihood
export expected_pi!, expected_pi
export load_model

end
