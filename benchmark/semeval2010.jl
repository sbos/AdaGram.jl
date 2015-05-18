require("src/word2vec.jl")

using word2vec

vm, dict = load_model(ARGS[1])
window = int(ARGS[2])

prepare_context(ctx) = map(int32, filter(v -> v > 0, map(w -> get(dict.word2id, lowercase(w), int32(-1)), split(ctx))))

while !eof(STDIN)
	word_full, ctx_num = split(readline())
	ctx_num = int(ctx_num)
	word = word_full[1:search(word_full, '.')-1]

	for i in 1:ctx_num
		left_ctx, right_ctx = split(readline(), '\t')

		left_ctx = prepare_context(left_ctx)
		right_ctx = prepare_context(right_ctx)

		ctx = vcat(right_ctx[1:min(length(right_ctx), window)], left_ctx[max(length(left_ctx)-window+1, 1):end])

		z = disambiguate(vm, dict.word2id[word], int32(ctx))
		s = indmax(z)
		println(word_full, " ", word_full, ".", i, " ", word_full, ".senses.", s)
	end
end
