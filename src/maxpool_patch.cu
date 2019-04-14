#include "stdafx.h"
#include <float.h>
#include "tensor.h"
struct maxpool_params {
	float* dst;
	float* src;
	int* indexes;
	int batch;
	int channels;
	int in_width;
	int in_height;
	int out_width;
	int out_height; 
	int window_w;
	int window_h; 
	int pad_w;
	int pad_h;
	int stride_w;
	int stride_h;
	bool is_nchw;
};

//NCHW only
__global__ void forward_maxpool_kernel(maxpool_params p , int elements, int threads) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;
 
	int w_offset = - p.pad_w * 0.5;
	int h_offset = - p.pad_h * 0.5;

	while (id < elements) {	
		int x = id % p.out_width;
		int temp = id / p.out_width;
		int y = temp % p.out_height;
		temp /= p.out_height;
		int c = temp % p.channels;
		int b = temp / p.channels;

		float max_v = -INFINITY;
		int max_i = -1;
 
		for (int i = 0; i < p.window_h; i++ ) {
			int src_y = h_offset + y * p.stride_h + i;
			for (int j = 0; j < p.window_w; j++) {				
				int src_x = w_offset + x * p.stride_w + j;
				int index = src_x + p.in_width *(src_y + p.in_height * (c + b * p.channels));
				bool valid = (src_y >= 0 && src_y < p.in_height && src_x >= 0 && src_x < p.in_width);
				float val = valid ? p.src[index] : -INFINITY;
				if (val > max_v) {
					max_i = index;
					max_v = val;
				}
			}
		}
		p.dst[id] = max_v;
		if (p.indexes) 
			p.indexes[id] = max_i;
		id += threads;
	}
}

__global__ void backward_maxpool_kernel(maxpool_params p, int elements, int threads) {
	
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int w_offset = - p.pad_w * 0.5;
	int h_offset = - p.pad_h * 0.5;

	int area_w = (p.window_w - 1) / p.stride_w;
	int area_h = (p.window_h - 1) / p.stride_h;

	float* prev_delta = p.dst;
	float* delta = p.src;

	while (id < elements) {
		int temp = id;
		int x = id % p.in_width;
		temp = id / p.in_width;
		int y = temp % p.in_height;
		temp /= p.in_height;
		int k = temp % p.channels;
		int b = temp / p.channels; 

		float d = 0.0f;
		for (int i = -area_h; i < area_h + 1; i++) {
			int src_y = (y - h_offset) / p.stride_h + i;
			for (int j = -area_w; j < area_w + 1; j++) {
				int src_x = (x - w_offset) / p.stride_w + j;				
				int index = src_x + p.out_width * (src_y + p.out_height * (k + p.channels * b));
				bool valid = (src_x >= 0 && src_x < p.out_width && src_y >= 0 && src_y < p.out_height);
				if(valid && p.indexes[index] == id)
					d += delta[index];
			}
		}
		prev_delta[id] = d;
		id += threads;
	}
}

bool forward_maxpool(FloatTensor4D& output, const FloatTensor4D& input, int* indexes, int window_w, int window_h, int pad_w, int pad_h, int stride_w, int stride_h) {
	if (output.MemElements() == 0 || input.MemElements() == 0) return false;

	int g = GPUGridSize();
	int b = GPUBlockSize();

	int threads = g * b;

	maxpool_params p =  {
		output.GetMem(),
		input.GetMem(),
		indexes,
		input.GetBatch(),
		input.GetChannels(),
		input.GetWidth(),
		input.GetHeight(),
		output.GetWidth(),
		output.GetHeight(),
		window_w,
		window_h,
		pad_w,
		pad_h,
		stride_w,
		stride_h,
		input.GetOrder() == TO_NCHW 
	};
	forward_maxpool_kernel<<<g,b>>>(p, output.MemElements(), threads);
	cudaError_t e = cudaDeviceSynchronize();
	if (e != cudaSuccess) {
		cerr << "e :" << (int)e << endl;
	}
	return e == cudaSuccess;
} 

bool backward_maxpool(FloatTensor4D& prev_delta, const FloatTensor4D& delta, int* indexes, int window_w, int window_h, int pad_w, int pad_h, int stride_w, int stride_h) {
	if (prev_delta.MemElements() == 0 || delta.MemElements() == 0) return false;
	int g = GPUGridSize();
	int b = GPUBlockSize();

	maxpool_params p = {
		prev_delta.GetMem(),
		delta.GetMem(),
		indexes,
		delta.GetBatch(),
		delta.GetChannels(),
		prev_delta.GetWidth(),
		prev_delta.GetHeight(),
		delta.GetWidth(),
		delta.GetHeight(),
		window_w,
		window_h,
		pad_w,
		pad_h,
		stride_w,
		stride_h,
		prev_delta.GetOrder() == TO_NCHW
	};

	int threads = g * b;
	backward_maxpool_kernel<<<g,b>>>(p, prev_delta.MemElements(), threads);
	cudaError_t e = cudaDeviceSynchronize();
	return e == cudaSuccess;
}

 