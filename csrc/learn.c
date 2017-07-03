#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

typedef float real;			// Precision of float numbers

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
	float lr, float sense_treshold) {

	--x;

	float pr = 0;

	for (int k = 0; k < T; ++k)
		for (int i = 0; i < M; ++i)
			in_grad[k*M + i] = 0;

	for (int n = 0; n < length && code[n] != -1; ++n) {
		float* out = Out + (path[n]-1)*M;

		for (int i = 0; i < M; ++i)
			out_grad[i] = 0;

		for (int k = 0; k < T; ++k) {
			if (z[k] < sense_treshold) continue;

			float* in = in_offset(In, x, k, M, T);

			float f = 0;
			for (int i = 0; i < M; ++i)
				f += in[i] * out[i];

			pr += z[k] * logsigmoid(f * (1 - 2*code[n]));

			float d = 1 - code[n] - sigmoid(f);
			float g = z[k] * lr * d;

			for (int i = 0; i < M; ++i) {
				in_grad[k*M + i] += g * out[i];
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

// Runs K-means on the word vectors; taken from the word2vec clustering routine
void kmeans(char** words, float* syn0, int classes, int vocab_size, 
  int layer1_size, char* outputFile){

  long a, b, c, d;
  FILE *fo;
  int clcn = classes, iter = 10, closeid;
  int *centcn = (int *)malloc(classes * sizeof(int));
  int *cl = (int *)calloc(vocab_size, sizeof(int));
  real closev, x;
  real *cent = (real *)calloc(classes * layer1_size, sizeof(real));

  fo = fopen(outputFile, "wb");

  for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
  for (a = 0; a < iter; a++) {
    for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
    for (b = 0; b < clcn; b++) centcn[b] = 1;
    for (c = 0; c < vocab_size; c++) {
      for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
      centcn[cl[c]]++;
    }
    for (b = 0; b < clcn; b++) {
      closev = 0;
      for (c = 0; c < layer1_size; c++) {
        cent[layer1_size * b + c] /= centcn[b];
        closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
      }
      closev = sqrt(closev);
      for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
    }
    for (c = 0; c < vocab_size; c++) {
      closev = -10;
      closeid = 0;
      for (d = 0; d < clcn; d++) {
        x = 0;
        for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
        if (x > closev) {
          closev = x;
          closeid = d;
        }
      }
      cl[c] = closeid;
    }
  }

  // Save the K-means classes
  for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", words[a], cl[a]);

  free(centcn);
  free(cent);
  free(cl);
  fclose(fo);
}