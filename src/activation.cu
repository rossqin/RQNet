#include "stdafx.h"
#include "activation.h"



__global__ static void activate_kernel(float* data, int elements, int threads, ACTIVATION_TYPE a) {

	float val;
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	while (index < elements) {

		switch (a) {
		case LEAKY:
			if (data[index] < 0.0)  data[index] *= 0.1;
			break;
		case LOGISTIC:
			//__device__ float logistic_activate_kernel(float x){return 1./(1. + exp(-x));}  
			data[index] = 1.0 / (1.0 + exp(-data[index]));
			break;
		case RELU:
			if (data[index] < 0) data[index] = 0.0;
			break;
		case HARDTAN:
			if (data[index] < -1.0)
				data[index] = -1.0;
			else if (data[index] > 1.0)
				data[index] = 1.0;
			break;
		case LHTAN:
			if (data[index] < -0.0)
				data[index] *= 0.001;
			else if (data[index] > 1.0)
				data[index] = 0.001 * (data[index] - 1) + 1;
			break;
		case TANH:
			val = exp(2.0 * data[index]);
			data[index] = (val - 1) / (val + 1);
			break;
		case LOGGY:
			data[index] = 2.0 / (1.0 + exp(-data[index]));
			break;
		case ELU:
			if (data[index] < 0)
				data[index] = (exp(data[index]) - 1);
			break;
		case RELIE:
			if (data[index] < 0)
				data[index] *= 0.01;
			break;
		case PLSE:
			if (data[index] < -4)
				data[index] = 0.01 * (data[index] + 4);
			else if (data[index] > 4)
				data[index] = 0.01 * (data[index] - 4) + 1.0;
			else
				data[index] = 0.125 * data[index] + 0.5;
			break;
		case RAMP:
			//return x*(x>0) + 0.1*x;
			val = 0.1 * data[index];
			if (data[index] > 0.0)
				data[index] += val;
			else
				data[index] = val;
			break;
		case STAIR:
		{
			int n = floor(data[index]);
			float t = (float)(n >> 1);
			if (0 == n & 0x01)
				data[index] = t; //if (n % 2 == 0) return floor(x / 2.);
			else
				data[index] = t + (data[index] - n);
			break;
		}
		case LINEAR:
		default:
			break;
		}
		index += threads;
	}

}
#define MIN_DELTA_VAL 1.0e-8
// output is delta
__global__ static void gradient_kernel(float* data, float* delta, int elements, int threads, ACTIVATION_TYPE a) {

	float val;
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	while (index < elements) {
		switch (a) {
		case LEAKY:
			if (data[index] < 0.0) delta[index] *= 0.1;
			break;
		case LOGISTIC:
			val = data[index] * (1.0 - data[index]);
			delta[index] *= val;
			break;
		case RELU:
			if (data[index] <= 0.0) delta[index] = 0.0;
			break;
		case HARDTAN:
			if (data[index] > -1.0 && data[index] < 1.0) delta[index] = 1.0;
			else delta[index] = 0.0;
			break;
		case LHTAN:
			if (data[index] <= 0.0 || data[index] >= 1.0)
				delta[index] *= 0.001;
			break;
		case TANH:
			val = data[index] * data[index];
			delta[index] *= (1.0 - val);
			break;
		case LOGGY:
			val = (data[index] + 1.0) * 0.5;
			delta[index] = 2.0 * (1 - val) * val * delta[index];
			break;
		case ELU:
			if (data[index] < 0.0)
				delta[index] *= (data[index] + 1.0);
			break;
		case RELIE:
			if (data[index] <= 0) delta[index] *= 0.01;
			break;
		case PLSE:
			if (data[index] < 0 || data[index] > 1)  delta[index] *= 0.01;
			else delta[index] *= 0.125;
			break;
		case RAMP:
			if (data[index] > 0) delta[index] *= 1.1;
			else
				delta[index] *= 0.1;
			break;
		case STAIR:
			if (floor(data[index]) == data[index]) delta[index] = 0.0;
		case LINEAR:
		default:
			break;
		}		
		if (delta[index] < MIN_DELTA_VAL && delta[index] > -MIN_DELTA_VAL)
			delta[index] = 0.0f;
		index += threads;
	}
}
static int dbg_index = 1;
bool gradient_array_ongpu(float *x, float * delta, int n, ACTIVATION_TYPE a) {
 
	if (a == LINEAR) return true;
	int g = GPUGridSize(9999);
	int b = GPUBlockSize(9999);
	int threads = g * b;
	if (n < threads) {
		b = (n + g - 1) / g;
		threads = g * b;
	}
#if 0
	char buffer[200];
	char *cpu_data = NULL;
	size_t bytes;
	ofstream f;
	if (a == LEAKY) { 
		bytes = n * sizeof(float);
		cpu_data = new char[bytes];
		cudaError_t err = cudaMemcpy(cpu_data, delta, bytes, cudaMemcpyDeviceToHost);
		sprintf(buffer, "E:\\AI\\Data\\debugging\\RQNet\\grdient.%02d.before.bin", dbg_index );
		f.open(buffer, ios::trunc); 
		f.write(cpu_data, bytes);
		f.close();
	}
#endif
	gradient_kernel <<<g, b >>>(x, delta , n, threads, a);
	cudaError_t err = cudaDeviceSynchronize();
#if 0
	if (a == LEAKY) {
		err = cudaMemcpy(cpu_data, delta, bytes, cudaMemcpyDeviceToHost);
		sprintf(buffer, "E:\\AI\\Data\\debugging\\RQNet\\grdient.%02d.after.bin", dbg_index++);
		f.open(buffer, ios::trunc);
		f.write(cpu_data, bytes);
		f.close();
		delete[]cpu_data;
	}
#endif
	if (err != cudaSuccess) {
		cerr << "activation failed!" << endl;
		return false;
	}
	return true;
}
bool print_debugging = false;
bool activate_array_ongpu(float *x, int n, ACTIVATION_TYPE a) {

	int g = GPUGridSize(9999);
	int b = GPUBlockSize(9999);
	int threads = g * b;
	if (n < threads) {
		b = (n + g - 1) / g;
		threads = g * b;
	}
#if 0
	char buffer[200];
	float *cpu_data = NULL;
	ofstream f;
	if (print_debugging) {
		sprintf(buffer, "activation array for 0x%08x , elments : %d, type: %d\n", (unsigned long)x, n, (int)a);
		cout << buffer;
		cpu_data = new float[n];
		cudaError_t err = cudaMemcpy(cpu_data, x, n * sizeof(float), cudaMemcpyDeviceToHost);

		sprintf(buffer, "activation.%08x.txt", (unsigned long)x);
		f.open(buffer, ios::trunc);

		for (int i = 0; i < n; i++) {
			sprintf(buffer, "%.6f ", cpu_data[i]);
			f << buffer;
		}
		f << endl;
	}
#endif
	activate_kernel <<<g, b>>>(x, n, threads, a);
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		cerr << "activation failed!" << endl;
		return false;
	}
#if 0
	if (print_debugging) {
		err = cudaMemcpy(cpu_data, x, n * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = 0; i < n; i++) {
			sprintf(buffer, "%.6f ", cpu_data[i]);
			f << buffer;
		}
		f << endl;
		f.close();
		delete[]cpu_data;
	}
#endif
	return true;
}