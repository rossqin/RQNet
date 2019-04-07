#include "stdafx.h"
#include <float.h>
#include "tensor.h"
struct one_stride_maxpool_params {
	float* dst;
	float* src;
	int* indexes;
	int batch;
	int channels;
	int width;
	int height;
	int window_w;
	int window_h; 
	bool is_nchw;
};
__global__ static void forward_one_stride_maxpool_kernel(one_stride_maxpool_params p) {
	
	for (int b = blockIdx.x; b < p.batch; b += gridDim.x) {
		if (p.is_nchw) {
			for (int c = threadIdx.x; c < p.channels; c += blockDim.x) {
				int base = b * p.channels * p.width * p.height +  c * p.width * p.height ;
				float* src = p.src + base;
				float* dst = p.dst + base;
				int* indexes = p.indexes + base;
				int idx_dst = 0;
				for (int y = 0; y < p.height; y++) {
					int window_top =  y ;
					int window_bottom = window_top + p.window_h;
					if (window_bottom >= p.height) window_bottom = p.height;
					for (int x = 0; x < p.width; x++, idx_dst++) {
						int window_left = x ;
						int window_right = window_left + p.window_w;
						if (window_right > p.width) window_right = p.width;
						float _max = -FLT_MAX; 
						int _max_i = 0;
						for (int i = window_top; i < window_bottom; i++) {
							for (int j = window_left; j < window_right; j++) {
								int idx_src = i * p.width + j;
								if (isnan(src[idx_src])) { // treat nan as 0
									if (src[idx_src] > _max) {
										_max = src[idx_src];
										_max_i = idx_src;
									}
								}
								if (src[idx_src] > _max) {
									_max = src[idx_src];
									_max_i = idx_src;
								}
							}
						}
						dst[idx_dst] = _max;
						indexes[idx_dst] = _max_i;
					}
				}
			}
		}
		else {
			int base = b * p.channels * p.width * p.height;
			float* src = p.src + base;
			float* dst = p.dst + base;
			int* indexes = p.indexes + base; 
			int idx_dst = 0;
			for (int y = 0; y < p.height; y++) {
				int window_top = y;
				int window_bottom = window_top + p.window_h;
				if (window_bottom >= p.height) window_bottom = p.height;
				for (int x = 0; x < p.width; x++, idx_dst += p.channels) {
					int window_left = x;
					int window_right = window_left + p.window_w;
					if (window_right > p.width) window_right = p.width;
					for (int c = threadIdx.x; c < p.channels; c += blockDim.x) {
						float _max = -FLT_MAX;
						int _max_i = 0;
						for (int i = window_top; i < window_bottom; i++) {
							for (int j = window_left; j < window_right; j++) {
								int idx_src = (i * p.width + j) * p.channels + c;
								if (isnan(p.src[idx_src])) { // treat nan as 0
									if (p.src[idx_src] > _max) {
										_max = p.src[idx_src];
										_max_i = idx_src;
									}
								}
								if (p.src[idx_src] > _max) {
									_max = p.src[idx_src];
									_max_i = idx_src;
								}
							}
						}
						p.dst[idx_dst + c] = _max;
						p.indexes[idx_dst + c] = _max_i;
					}
				}
			}
		}
	}
}

bool forward_one_stride_maxpool(FloatTensor4D& output, const FloatTensor4D& input, int* indexes, int window_w, int window_h) {
	if (output.MemElements() == 0 ||
		output.MemElements() != input.MemElements() ||
		NULL == indexes)
		return false;
	int g = GPUGridSize(input.GetBatch());
	int b = GPUBlockSize(input.GetChannels());
	
	one_stride_maxpool_params p = {
		output.GetMem(),
		input.GetMem(),
		indexes,
		input.GetBatch(),
		input.GetChannels(),
		input.GetWidth(),
		input.GetHeight(),
		window_w,
		window_h,
		input.GetOrder() == TO_NCHW 
	};
	forward_one_stride_maxpool_kernel<<<g,b>>>(p);
	cudaError_t e = cudaDeviceSynchronize();
	return e == cudaSuccess;
}
__global__ static void backward_one_stride_maxpool_kernel(one_stride_maxpool_params p) {
	//int c_elements = p.width * p.height;
	for (int b = blockIdx.x; b < p.batch; b += gridDim.x) {
		if (p.is_nchw) {
			for (int c = threadIdx.x; c < p.channels; c += blockDim.x) {
				int base = b * p.channels * p.width * p.height + c * p.width * p.height;
				float* src = p.src + base;
				float* dst = p.dst + base;
				int* indexes = p.indexes + base;
				int index = 0;
				for (int y = 0; y < p.height; y++) { 
					for (int x = 0; x < p.width; x++, index++) {
						dst[index] = (indexes[index] == index) ? src[index] : 0.0f;
					}
				}
			}
		}
		else {
			int base = b * p.channels * p.width * p.height;
			float* src = p.src + base;
			float* dst = p.dst + base;
			int* indexes = p.indexes + base;
			int index = 0;
			for (int y = 0; y < p.height; y++) { 
				for (int x = 0; x < p.width; x++, index += p.channels) {					 
					for (int c = threadIdx.x; c < p.channels; c += blockDim.x) { 
						int i = index + c;
						dst[i] = (indexes[i] == i) ? src[i] : 0.0f;
					}
				}
			}
		}
	}
}
bool backward_one_stride_maxpool(FloatTensor4D& delta, int* indexes) {
	float* delta_cpy = NULL;
	if (delta.MemElements() == 0 || NULL == indexes)
		return false;
	if (cudaSuccess != cudaMalloc(&delta_cpy, delta.MemBytes())) return false;

	if (cudaSuccess != cudaMemcpy(delta_cpy, delta.GetMem(), delta.MemBytes(), cudaMemcpyDeviceToDevice)) {
		cudaFree(delta_cpy);
		return false;
	}

	int g = GPUGridSize(delta.GetBatch());
	int b = GPUBlockSize(delta.GetChannels());

	one_stride_maxpool_params p = {
		delta.GetMem(),
		delta_cpy,		
		indexes,
		delta.GetBatch(),
		delta.GetChannels(),
		delta.GetWidth(),
		delta.GetHeight(),
		2,
		2,
		delta.GetOrder() == TO_NCHW
	};
	backward_one_stride_maxpool_kernel<<<g,b>>>(p);
	cudaError_t e = cudaDeviceSynchronize();
	cudaFree(delta_cpy);
	return e == cudaSuccess;
}
/*
struct one_stride_maxpool_params {
float* dst;
float* src;
int* indexes;
int batch;
int channels;
int width;
int height;
int window_w;
int window_h;
bool is_nchw;
};
*/