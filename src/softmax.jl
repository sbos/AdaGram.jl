using Base.Collections
using Base.Order
using Devectorize

type HierarchicalSoftmaxNode
	parent::Int32
	branch::Bool
end

type HierarchicalOutput
	code::Array{Int8}
	path::Array{Int}
end

import Base.length

length(out::HierarchicalOutput) = length(out.path)

function HierarchicalSoftmaxNode() 
	return HierarchicalSoftmaxNode(int32(0), false)
end

function softmax_path(nodes::Array{HierarchicalSoftmaxNode}, 
		V::Integer, id::Integer)
	function path()
		while true
			node = nodes[id]
			if node.parent == 0 break; end
			@assert node.parent > V
			produce((int32(node.parent - V), convert(Float64, node.branch)))
			id = node.parent
		end
	end

	return Task(path)
end

function build_huffman_tree{Tf <: Number}(freqs::Array{Tf})
	V = length(freqs)
	nodes = Array(HierarchicalSoftmaxNode, V)
	for v in 1:V
		nodes[v] = HierarchicalSoftmaxNode()
	end

	freq_ord = By(wf -> wf[2])
	heap = heapify!([(nodes[v], freqs[v]) for v in 1:V], freq_ord)

	function pop_initialize!(parent::Int, branch::Bool)
		node = heappop!(heap, freq_ord)
		node[1].parent = int32(parent)
		node[1].branch = branch
		return node[2]
	end

	L = V
	while length(heap) > 1
		L += 1
		node = HierarchicalSoftmaxNode()
		push!(nodes, node)

		freq = 1
		freq = pop_initialize!(L, true) + pop_initialize!(L, false)
		heappush!(heap, (node, freq), freq_ord)
	end

	@assert length(heap) == 1

	return nodes
end

function convert_huffman_tree(nodes::Array{HierarchicalSoftmaxNode}, V::Integer)
	outputs = Array(HierarchicalOutput, V)
	for v in 1:V
		code = Array(Int8, 0)
		path = Array(Int, 0)

		for (n, branch) in softmax_path(nodes, V, v)
			push!(code, uint8(branch))
			push!(path, n)
		end

		outputs[v] = HierarchicalOutput(code, path)
	end

	return outputs
end

export HierarchicalSoftmaxNode, softmax_path