using AdaGram

vm, dict = load_model(ARGS[1])
f = open(ARGS[2], "w")

z = zeros(T(vm))
for v in 1:V(vm)
	expected_pi!(z, vm, v)
	for k in 1:T(vm)
		if z[k] < 5e-3 continue end
		if dict.id2word[v] != "*UNKNOWN*"
		print(f, dict.id2word[v], "_", k)
		else print(f, dict.id2word[v])
		end
		for i in 1:M(vm)
			print(f, " ", vm.In[i, k, v])
		end
		println(f)
	end	
end

close(f)
