#include "stdafx.h"
#include "activation.h"
#include <cuda_fp16.h>

__global__ static void activate_float_kernel(const float* in, float* out, int elements,
	cudnnDataType_t data_type, ActivationMode mode) { 
	float val;
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	int threads = gridDim.x * blockDim.x;
	while (index < elements) {
		switch (mode) {
		case LEAKY:
			if (in[index] < 0.0)
				out[index] = 0.1f * in[index];
			else
				out[index] = in[index];
			break;
		case LOGISTIC:
			out[index] = 1.0 / (1.0 + exp(-in[index]));

			break;
		case RELU:
			if (in[index] < 0.0)
				out[index] = 0.0;
			else
				out[index] = in[index];
			break;
		case HARDTAN:
			if (in[index] < -1.0)
				out[index] = -1.0;
			else if (in[index] > 1.0)
				out[index] = 1.0;
			else
				out[index] = in[index];
			break;
		case LHTAN:

			if (in[index] < 0.0f)
				out[index] = in[index] * 0.001f;
			else if (in[index] > 1.0f)
				out[index] = 0.001f * (in[index] - 1.0f) + 1.0f;
			else
				out[index] = in[index];

			break;
		case TANH:
			val = exp(2.0f * in[index]);
			out[index] = (val - 1.0f) / (val + 1.0f);
			break;
		case LOGGY:
			val = 2.0f / (1.0f + exp(-in[index]));
			out[index] = 2.0f / (1.0f + exp(-in[index]));
			break;
		case ELU:
			if (in[index] < 0.0f)
				out[index] = exp(in[index]) - 1.0f;
			else
				out[index] = in[index];
			break;

		case LINEAR:
			out[index] = in[index];
			break;
		default:
			break;
		}
		index += threads;
	}

}
__global__ static void activate_half_kernel(const __half* in, __half* out, int elements,
	cudnnDataType_t data_type, ActivationMode mode) {


	float val;
	float fx;
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	int threads = gridDim.x * blockDim.x;
	while (index < elements) {
		fx = __half2float(in[index]);
		switch (mode) {
		case LEAKY:
			if (fx < 0.0f)
				out[index] = __float2half(fx * 0.1);
			else
				out[index] = in[index];

			break;
		case LOGISTIC:
			val = 1.0f / (1.0f + exp(-fx));
			out[index] = __float2half(val);
			break;
		case RELU:
			if (fx < 0.0f) {
				out[index] = 0.0;
			}
			else
				out[index] = in[index];

			break;
		case HARDTAN:
			 
			if (fx < -1.0f)
				out[index] = __float2half(-1.0f);
			else if (val > 1.0f)
				out[index] = __float2half(1.0f);
			else
				out[index] = in[index]; 
			break;
		case LHTAN: 
			if (fx < 0.0f)
				out[index] = __float2half(fx * 0.001f);
			else if (fx > 1.0f)
				out[index] = __float2half(0.001f * (fx - 1.0f) + 1.0f);
			else
				out[index] = in[index]; 
			break;
		case TANH: 
			val = exp(2.0f * fx);
			val = (val - 1.0f) / (val + 1.0f);
			out[index] = __float2half(val); 
			break;
		case LOGGY: 
			val = 2.0f / (1.0f + exp(-fx));
			out[index] = __float2half(val); 
			break;
		case ELU: 
			if (fx < 0.0f) {
				out[index] = __float2half(exp(fx) - 1.0f); 
			}
			else
				out[index] = in[index];
			break;

		case LINEAR:
			out[index] = in[index];
			break;
		default:
			break;
		}
		index += threads;
	}

} 
// output is delta
__global__ static void gradient_float_kernel(const float* y, float* dy, int elements,
	cudnnDataType_t data_type, ActivationMode mode) {
	float val;
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	int threads = gridDim.x * blockDim.x;
	while (index < elements) {
		switch (mode) {
		case LEAKY:
			if (y[index] < 0.0) dy[index] *= 0.1f;
			break;
		case LOGISTIC:
			val = y[index] * (1.0 - y[index]);
			dy[index] *= val;
			break;
		case RELU:
			if (y[index] <= 0.0) dy[index] = 0.0;
			break;
		case HARDTAN:
			if (y[index] > -1.0 && y[index] < 1.0) dy[index] = 1.0;
			else dy[index] = 0.0;
			break;
		case LHTAN:
			if (y[index] <= 0.0 || y[index] >= 1.0)
				dy[index] *= 0.001;
			break;
		case TANH:
			val = y[index] * y[index];
			dy[index] *= (1.0 - val);
			break;
		case LOGGY:
			val = (y[index] + 1.0) * 0.5;
			dy[index] = 2.0 * (1 - val) * val * dy[index];
			break;
		case ELU:
			if (y[index] < 0.0)
				dy[index] *= (y[index] + 1.0);
			break;
		case LINEAR:
		default:
			break;
		}
		index += threads;
	}
}
__global__ static void gradient_half_kernel(const __half* y, __half* dy, int elements,
	cudnnDataType_t data_type, ActivationMode mode) {
	float val;
	float fy;
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	int threads = gridDim.x * blockDim.x;
	while (index < elements) {
		fy = __half2float(y[index]);
		switch (mode) {
		case LEAKY:
			if (fy < 0.0f) dy[index] = __hmul(dy[index] , __float2half(0.1f));
			break;
		case LOGISTIC:			
			val = fy * (1.0f - fy) * __half2float(dy[index]);
			dy[index] = __float2half(val);
			break;
		case RELU:
			if (fy <= 0.0f) dy[index] = __float2half(0.0f);
			break;
		case HARDTAN:
			if (fy > -1.0f && fy< 1.0f) dy[index] = __float2half(1.0f);
			else dy[index] = __float2half(0.0f);
			break;
		case LHTAN:
			if (fy <= 0.0f || fy >= 1.0)
				dy[index] = __hmul(dy[index], __float2half(0.001f));
			break;
		case TANH:
			val = 1.0f - fy * fy;
			dy[index] = __hmul(dy[index], __float2half(val));
			break;
		case LOGGY:{
			val = (fy + 1.0f) * 0.5f;
			float temp = 2.0f * (1.0f - val) * val;
			dy[index] = __hmul(dy[index], __float2half(temp));
			}
			break;
		case ELU:
			if (fy < 0.0f) {
				val = fy + 1.0f;
				dy[index] = __hmul(dy[index], __float2half(val));
			}
			break; 
		default:
			break;
		}
		index += threads;
	}
}
bool gradient_array_ongpu(const void* y, void* delta, int elements, cudnnDataType_t data_type, ActivationMode mode) {
	if (mode == LINEAR) return true;

	int g = GPUGridSize();
	int b = GPUBlockSize(); 
	
	if (data_type == CUDNN_DATA_FLOAT) {
		const float* in = reinterpret_cast<const float*>(y);
		float* out = reinterpret_cast<float*>(delta);
		gradient_float_kernel <<<g, b >>> (in, out, elements, data_type, mode);
	}
	else if (data_type == CUDNN_DATA_HALF) {
		const __half* in = reinterpret_cast<const __half*>(y);
		__half* out = reinterpret_cast<__half*>(delta);
		gradient_half_kernel <<<g, b >>> (in, out, elements, data_type, mode);
	}
	else {
		return false;
	}
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		cerr << "gradient_array_ongpu failed! elements:" << elements << ", err :" << (int)err << endl;
		return false;
	}
	return true;
}
 
bool activate_array_ongpu(const void* x, void* y, int elements, cudnnDataType_t data_type, ActivationMode mode) {

	int g = GPUGridSize();
	int b = GPUBlockSize(); 
	if (data_type == CUDNN_DATA_FLOAT) {
		const float* in = reinterpret_cast<const float*>(x);
		float* out = reinterpret_cast<float*>(y);
		if (mode == LINEAR)
			return cudaSuccess == cudaMemcpy(out, in, elements * sizeof(float), cudaMemcpyDeviceToDevice);

		activate_float_kernel <<<g, b >>> (in, out, elements, data_type, mode);
	}
	else if (data_type == CUDNN_DATA_HALF) {
		const __half* in = reinterpret_cast<const __half*>(x);
		__half* out = reinterpret_cast<__half*>(y);
		if (mode == LINEAR)
			return cudaSuccess == cudaMemcpy(out, in, elements * sizeof(__half), cudaMemcpyDeviceToDevice);
		activate_half_kernel <<<g, b >>> (in, out, elements, data_type, mode);
	}
	else {
		return false;
	}
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		cerr << "activate_array_ongpu failed! elements:" << elements << ", err :" << (int)err << endl;
		return false;
	}
	return true;
}