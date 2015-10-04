const MAX_VAL = 6f0
const TABLE_SIZE = 1000

sigmoid_table = zeros(Tsf, TABLE_SIZE)
logsigm_table = zeros(Tsf, TABLE_SIZE)
for i in 1:TABLE_SIZE
	sigmoid_table[i] = exp(((i-1) / TABLE_SIZE * 2 - 1) * MAX_VAL)
	sigmoid_table[i] = sigmoid_table[i] / (1. + sigmoid_table[i])
	logsigm_table[i] = log(sigmoid_table[i] + realmin(Tsf))
end

function index(x::Tsf)
	return round((x + MAX_VAL) * (TABLE_SIZE / MAX_VAL / 2) ) + 1
end

function cached_sigmoid(x::Tsf)
	if x >= MAX_VAL
		return 1f0
	elseif x <= -MAX_VAL
		return 0f0
	end

	idx = index(x)
	if idx > TABLE_SIZE
		return 1f0
	end
	return sigmoid_table[idx]
end

function cached_logsigm(x::Tsf)
	if x >= MAX_VAL
		return 0f0
	elseif x <= -MAX_VAL
		return -6.01f0
	end

	idx = index(x)
	if idx > TABLE_SIZE
		return 0f0
	end
	return logsigm_table[idx]
end
