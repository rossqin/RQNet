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
	int area = p.width * p.height;
	for (int b = blockIdx.x; b < p.batch; b += gridDim.x) {
		int b_off = b * p.channels * area;
		if (p.is_nchw) {
			for (int c = threadIdx.x; c < p.channels; c += blockDim.x) {
				int c_off = b_off +  c * area;
				float* src = p.src + c_off;
				float* dst = p.dst + c_off;
				int* indexes = p.indexes + c_off;
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
								if (isnan(src[idx_src]) || isinf(src[idx_src])) { // treat nan as 0
									if (0.0f > _max) {
										_max = 0.0f;
										_max_i = idx_src;
									}
								}
								else if (src[idx_src] > _max) {
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
								if (isnan(src[idx_src]) || isinf(src[idx_src])) { // treat nan as 0
									if (0.0f > _max) {
										_max = 0.0f;
										_max_i = idx_src;
									}
								}
								else if (src[idx_src] > _max) {
									_max = src[idx_src];
									_max_i = idx_src;
								}
							}
						}
						dst[idx_dst + c] = _max;
						indexes[idx_dst + c] = _max_i;
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
				int offset = b * p.channels * p.width * p.height + c * p.width * p.height;	 
				float* dst = p.dst + offset;
				int* indexes = p.indexes + offset;
				int index = 0;
				for (int y = 0; y < p.height; y++) { 
					for (int x = 0; x < p.width; x++, index++) {
						if(indexes[index] != index) 
							dst[index] = 0.0f;
					}
				}
			}
		}
		else {
			int base = b * p.channels * p.width * p.height; 
			float* dst = p.dst + base;
			int* indexes = p.indexes + base;
			int index = 0;
			for (int y = 0; y < p.height; y++) { 
				for (int x = 0; x < p.width; x++, index += p.channels) {					 
					for (int c = threadIdx.x; c < p.channels; c += blockDim.x) { 
						int i = index + c;
						if(indexes[i] != i) 
							dst[i] = 0.0f;
					}
				}
			}
		}
	}
}
bool backward_one_stride_maxpool(FloatTensor4D& delta, int* indexes) {
	if (delta.MemElements() == 0 || NULL == indexes)
		return false;
	 
	int g = GPUGridSize(delta.GetBatch());
	int b = GPUBlockSize(delta.GetChannels());

	one_stride_maxpool_params p = {
		delta.GetMem(),
		NULL,		
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