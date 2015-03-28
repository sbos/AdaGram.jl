#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

float sigmoid(float x) {
	return 1 / (1 + exp(-x));
}

float logsigmoid(float x) {
	return -log(1 + exp(-x));
}

#define in_offset(In, x, k, M, T) (In) + (x)*(M)*(T) + (k)*(M)

typedef long long int Int;

//assuming everything is indexed from 1 like in julia
float inplace_update(float* In, float* Out, 
	Int M, Int T, double* z,
	Int x,  
	int32_t* path, int8_t* code, int64_t length,
	float* in_grad, float* out_grad, 
	float lr, float sense_treshold, float l2) {

	--x;

	float pr = 0;

	for (int k = 0; k < T; ++k)
		for (int i = 0; i < M; ++i)
			in_grad[k*M + i] = 0;

	for (int n = 0; n < length && code[n] != -1; ++n) {
		float* out = Out + (path[n]-1)*M;

		for (int i = 0; i < M; ++i)
			out_grad[i] = -l2 * lr * out[i];

		for (int k = 0; k < T; ++k) {
			if (z[k] < sense_treshold) continue;

			float* in = in_offset(In, x, k, M, T);

			float f = 0;
			for (int i = 0; i < M; ++i)
				f += in[i] * out[i];

			pr += z[k] * logsigmoid(f * (1 - 2*code[n]));

			float d = 1 - code[n] - sigmoid(f);
			float g = z[k] * lr * d;

			float in_l2  = l2 * z[k] * lr * x_freq / total_freq;

			for (int i = 0; i < M; ++i) {
				in_grad[k*M + i] += g * out[i] - l2 * z[k] * lr * in[i];
				out_grad[i]      += g * in[i];
			}
		}

		for (int i = 0; i < M; ++i)
			out[i] += out_grad[i];
	}

	for (int k = 0; k < T; ++k) {
		if (z[k] < sense_treshold) continue;
		float* in = in_offset(In, x, k, M, T);
		for (int i = 0; i < M; ++i)
			in[i] += in_grad[k*M + i];
	}

	return pr;
}

float skip_gram(float* In, float* Out, 
	Int M, 
	int32_t* path, int8_t* code, int length) {

	float pr = 0;

	for (int n = 0; n < length && code[n] != -1; ++n) {
		float* out = Out + (path[n]-1)*M;

		float f = 0;
		for (int i = 0; i < M; ++i)
			f += In[i] * out[i];

		pr += logsigmoid(f * (1 - 2*code[n]));
	}

	return pr;
}

void update_z(float* In, float* Out, 
	Int M, Int T, double* z,
	Int x,  
	int32_t* path, int8_t* code, int64_t length) {

	--x;

	for (int n = 0; n < length && code[n] != -1; ++n) {
		float* out = Out + (path[n]-1)*M;

		for (int k = 0; k < T; ++k) {
			float* in = in_offset(In, x, k, M, T);

			float f = 0;
			for (int i = 0; i < M; ++i)
				f += in[i] * out[i];

			z[k] += logsigmoid(f * (1 - 2*code[n]));
		}
	}
}
