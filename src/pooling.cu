#include "stdafx.h"
#include "cuda_tensor.h"

__global__ static void forward_maxpool_kernel_nchw(float* out, const float* in, int* indexes, int elements,
	int channels, int width, int height, int window, int stride, int pad) {

	int dst_i = blockIdx.x * blockDim.x + threadIdx.x;
	int threads = gridDim.x * blockDim.x;
	int dst_size = width * height;
	int dst_c_size = channels * dst_size;
	int stride_w = stride & 0xffff;
	int stride_h = stride >> 16;
	int window_w = window & 0xffff;
	int window_h = window >> 16; 
	
	int pad_hb = pad & 0xff;
	int pad_ht = (pad >> 8) & 0xff;
	int pad_wr = (pad >> 16) & 0xff;
	int pad_wl = pad >> 24;
	//output_width = (input_width + pad_wl + pad_wr - window_w) / stride_w + 1;
	int src_w = ( width - 1) * stride_w + window_w - pad_wl - pad_wr;
	//output_height = (input_height + pad_ht + pad_hb - window_h) / stride_h + 1;
	int src_h = (height - 1) * stride_h + window_h - pad_ht - pad_hb;
	
	int src_size = src_w * src_h;

	while (dst_i < elements) {
		int b = dst_i / dst_c_size;
		int temp = dst_i %  dst_c_size;
		int c = temp / dst_size;
		temp = temp % dst_size;
		int dst_y = temp / width;
		int dst_x = temp % width;
		out[dst_i] = -INFINITY;
		 
		for (int y = dst_y * stride_h - pad_ht, i = 0; i < window_h; i++,y++) {
			if (y >= 0 && y < src_h) {
				for (int x = dst_x * stride_w - pad_wl, j = 0; j < window_w; j++, x++) {
					if (x >= 0 && x < src_w) {
						int src_i = (b * channels + c) * src_size + y * src_w + x;
						if (in[src_i] > out[dst_i]) {
							out[dst_i] = in[src_i];
							indexes[dst_i] = src_i;
						}
					}
				}
			}
		}

		dst_i += threads;
	}

}

__global__ static void forward_avgpool_kernel_nchw(float* out, const float* in, int elements,
	int channels, int width, int height, int window, int stride, int pad) {

	int dst_i = blockIdx.x * blockDim.x + threadIdx.x;
	int threads = gridDim.x * blockDim.x;
	int dst_size = width * height;
	int dst_c_size = channels * dst_size;
	int stride_w = stride & 0xffff;
	int stride_h = stride >> 16;
	int window_w = window & 0xffff;
	int window_h = window >> 16;

	int pad_hb = pad & 0xff;
	int pad_ht = (pad >> 8) & 0xff;
	int pad_wr = (pad >> 16) & 0xff;
	int pad_wl = pad >> 24;
	//output_width = (input_width + pad_wl + pad_wr - window_w) / stride_w + 1;
	int src_w = (width - 1) * stride_w + window_w - pad_wl - pad_wr;
	//output_height = (input_height + pad_ht + pad_hb - window_h) / stride_h + 1;
	int src_h = (height - 1) * stride_h + window_h - pad_ht - pad_hb;

	int src_size = src_w * src_h;

	while (dst_i < elements) {
		int b = dst_i / dst_c_size;
		int temp = dst_i % dst_c_size;
		int c = temp / dst_size;
		temp = temp % dst_size;
		int dst_y = temp / width;
		int dst_x = temp % width;
		out[dst_i] = 0.0f;
		

		for (int y = dst_y * stride_h - pad_ht, i = 0; i < window_h; i++, y++) {
			if (y >= 0 && y < src_h) {
				for (int x = dst_x * stride_w - pad_wl, j = 0; j < window_w; j++, x++) {
					if (x >= 0 && x < src_w) {
						int src_i = (b * channels + c) * src_size + y * src_w + x;
						out[dst_i] += in[src_i]; 
					}
				}
			}
		}
		out[dst_i] /= src_size;
		dst_i += threads;
	}

}

bool forward_avgpool(CudaTensor& output, const CudaTensor& input, int window, int stride, int pad) {
	if (input.DataType() == CUDNN_DATA_HALF) {
		CudaPtr<float> in(input.Elements());
		if (!f16_to_f32(in, reinterpret_cast<__half*>(input.Data()), input.Elements())) {
			cudaFree(in);
			return false;
		}
		CudaPtr<float> out(output.Elements());
		int g = GPUGridSize();
		int b = GPUBlockSize();
		if (input.DataFormat() == CUDNN_TENSOR_NCHW) {
			forward_avgpool_kernel_nchw <<<g, b >>> (out, in, output.Elements(), output.Channel(),
				output.Width(), output.Height(), window, stride, pad);
		}
		else {
			//TODO: finish
			return false;
		}
		cudaError_t e = cudaDeviceSynchronize();
		if (e != cudaSuccess) {
			cerr << " Error: forward_maxpool failed!\n";
			return false;
		}
		return f32_to_f16(reinterpret_cast<__half*>(output.Data()), out, output.Elements());
	}
	else {
		int g = GPUGridSize();
		int b = GPUBlockSize();
		float* out = reinterpret_cast<float*>(output.Data());
		float* in = reinterpret_cast<float*>(input.Data());
		if (input.DataFormat() == CUDNN_TENSOR_NCHW) {
			forward_avgpool_kernel_nchw <<<g, b >>> (out, in,  output.Elements(), output.Channel(),
				output.Width(), output.Height(), window, stride, pad);
		}
		else {
			//TODO: finish
			return false;
		}
		cudaError_t e = cudaDeviceSynchronize();
		if (e != cudaSuccess) {
			cerr << " Error: forward_maxpool failed!\n";
			return false;
		}
	}
	return true;
}
bool forward_maxpool(CudaTensor& output, const CudaTensor& input, int* indexes,
	int window, int stride, int pad) {
	
	if (input.DataType() == CUDNN_DATA_HALF) {
		CudaPtr<float> in(input.Elements()); 
		if (!f16_to_f32(in, reinterpret_cast<__half*>(input.Data()), input.Elements())) {
			cudaFree(in);
			return false;
		}
		CudaPtr<float> out(output.Elements()); 
		int g = GPUGridSize();
		int b = GPUBlockSize();
		if (input.DataFormat() == CUDNN_TENSOR_NCHW) {
			forward_maxpool_kernel_nchw<<<g,b>>>(out, in, indexes, output.Elements(), output.Channel(), 
				output.Width(), output.Height(), window, stride, pad);
		}
		else {
			//TODO: finish
			return false;
		}
		cudaError_t e = cudaDeviceSynchronize();
		if (e != cudaSuccess) {
			cerr << " Error: forward_maxpool failed!\n";
			return false;
		}
		return f32_to_f16(reinterpret_cast<__half*>(output.Data()), out, output.Elements());
	}
	else {
		int g = GPUGridSize();
		int b = GPUBlockSize();
		float* out = reinterpret_cast<float*>(output.Data());
		float* in = reinterpret_cast<float*>(input.Data());
		if (input.DataFormat() == CUDNN_TENSOR_NCHW) {
			forward_maxpool_kernel_nchw<<<g, b>>>(out, in, indexes, output.Elements(), output.Channel(),
				output.Width(), output.Height(), window, stride, pad);
		}
		else {
			//TODO: finish
			return false;
		}
		cudaError_t e = cudaDeviceSynchronize();
		if (e != cudaSuccess) {
			cerr << " Error: forward_maxpool failed!\n";
			return false;
		}
	}
	return true;
}
__global__ static void backward_maxpool_kernel_nchw(float* out, const float* in, int* indexes, int elements,
	int channels, int width, int height, int window, int stride, int pad) {
	int dst_i = blockIdx.x * blockDim.x + threadIdx.x;
	int threads = gridDim.x * blockDim.x;
	int dst_size = width * height;
	int dst_c_size = channels * dst_size;
	int stride_w = stride & 0xffff;
	int stride_h = stride >> 16;
	int window_w = window & 0xffff;
	int window_h = window >> 16;

	int pad_hb = pad & 0xff;
	int pad_ht = (pad >> 8) & 0xff;
	int pad_wr = (pad >> 16) & 0xff;
	int pad_wl = pad >> 24; 
	int src_w = (width + pad_wl + pad_wr - window_w) / stride_w + 1;
	int src_h = (height + pad_ht + pad_hb - window_h) / stride_h + 1;
	int src_size = src_w * src_h;

	while (dst_i < elements) {
		int b = dst_i / dst_c_size;
		int temp = dst_i %  dst_c_size;
		int c = temp / dst_size;
		temp = temp % dst_size;
		int dst_y = temp / width;
		int dst_x = temp % width;

		int src_y = (dst_y + pad_ht) / stride_h;
		int src_x = (dst_x + pad_wl) / stride_w;
		//TODO: makesure src_x and src_y is in the matrix

		int src_i = (b * channels + c) * src_size + src_y * src_w + src_x;

		if (indexes[src_i] == dst_i)
			out[dst_i] += in[src_i]; 

		dst_i += threads;
	}
}
bool backward_maxpool(CudaTensor& dx, const CudaTensor& dy, int* indexes,
	int window, int stride, int pad) {
	dx = 0.0f;
	if (dx.DataType() == CUDNN_DATA_HALF) {
		CudaPtr<float> in(dy.Elements()); 
		if (f16_to_f32(in, reinterpret_cast<__half*>(dy.Data()), dy.Elements())) {
			cudaFree(in);
			return false;
		}
		CudaPtr<float> out(dx.Elements()); 
		int g = GPUGridSize();
		int b = GPUBlockSize();
		if (dy.DataFormat() == CUDNN_TENSOR_NCHW) {
			backward_maxpool_kernel_nchw<<<g,b>>>(out, in, indexes, dx.Elements(), dx.Channel(),
				dx.Width(), dx.Height(), window, stride, pad);
		}
		else {
			//TODO: finish
			return false;
		}
		cudaError_t e = cudaDeviceSynchronize();
		if (e != cudaSuccess) {
			cerr << " Error: forward_maxpool failed!\n";
			return false;
		}
		return f32_to_f16(reinterpret_cast<__half*>(dx.Data()), out, dx.Elements());
	}
	else {
		int g = GPUGridSize();
		int b = GPUBlockSize();
		float* out = reinterpret_cast<float*>(dx.Data());
		float* in = reinterpret_cast<float*>(dy.Data());
		if (dy.DataFormat() == CUDNN_TENSOR_NCHW) {
			backward_maxpool_kernel_nchw<<<g,b>>>(out, in, indexes, dx.Elements(), dx.Channel(), 
				dx.Width(), dx.Height(), window, stride, pad);
		}
		else {
			//TODO: finish
			return false;
		}
		cudaError_t e = cudaDeviceSynchronize();
		if (e != cudaSuccess) {
			cerr << " Error: forward_maxpool failed!\n";
			return false;
		}
	}
	return true;
}


__global__ static void backward_avgpool_kernel_nchw(float* out, const float* in, int elements,
	int channels, int width, int height, int window, int stride, int pad) {
	int dst_i = blockIdx.x * blockDim.x + threadIdx.x;
	int threads = gridDim.x * blockDim.x;
	int dst_size = width * height;
	int dst_c_size = channels * dst_size;
	int stride_w = stride & 0xffff;
	int stride_h = stride >> 16;
	int window_w = window & 0xffff;
	int window_h = window >> 16;

	int pad_hb = pad & 0xff;
	int pad_ht = (pad >> 8) & 0xff;
	int pad_wr = (pad >> 16) & 0xff;
	int pad_wl = pad >> 24;
	int src_w = (width + pad_wl + pad_wr - window_w) / stride_w + 1;
	int src_h = (height + pad_ht + pad_hb - window_h) / stride_h + 1;
	int src_size = src_w * src_h;
	int w_size = stride_w * stride_h;

	while (dst_i < elements) {
		int b = dst_i / dst_c_size;
		int temp = dst_i % dst_c_size;
		int c = temp / dst_size;
		temp = temp % dst_size;
		int dst_y = temp / width;
		int dst_x = temp % width;

		int src_y = (dst_y + pad_ht) / stride_h;
		int src_x = (dst_x + pad_wl) / stride_w;
		//TODO: makesure src_x and src_y is in the matrix

		int src_i = (b * channels + c) * src_size + src_y * src_w + src_x;

		out[dst_i] += in[src_i] / w_size;

		dst_i += threads;
	}
}
bool backward_avgpool(CudaTensor& dx, const CudaTensor& dy, int window, int stride, int pad) {
	dx = 0.0f;
	if (dx.DataType() == CUDNN_DATA_HALF) {
		CudaPtr<float> in(dy.Elements());
		if (f16_to_f32(in, reinterpret_cast<__half*>(dy.Data()), dy.Elements())) {
			cudaFree(in);
			return false;
		}
		CudaPtr<float> out(dx.Elements());
		int g = GPUGridSize();
		int b = GPUBlockSize();
		if (dy.DataFormat() == CUDNN_TENSOR_NCHW) {
			backward_avgpool_kernel_nchw <<<g, b >>> (out, in, dx.Elements(), dx.Channel(),
				dx.Width(), dx.Height(), window, stride, pad);
		}
		else {
			//TODO: finish
			return false;
		}
		cudaError_t e = cudaDeviceSynchronize();
		if (e != cudaSuccess) {
			cerr << " Error: forward_avgpool failed!\n";
			return false;
		}
		return f32_to_f16(reinterpret_cast<__half*>(dx.Data()), out, dx.Elements());
	}
	else {
		int g = GPUGridSize();
		int b = GPUBlockSize();
		float* out = reinterpret_cast<float*>(dx.Data());
		float* in = reinterpret_cast<float*>(dy.Data());
		if (dy.DataFormat() == CUDNN_TENSOR_NCHW) {
			backward_avgpool_kernel_nchw <<<g, b >>> (out, in, dx.Elements(), dx.Channel(),
				dx.Width(), dx.Height(), window, stride, pad);
		}
		else {
			//TODO: finish
			return false;
		}
		cudaError_t e = cudaDeviceSynchronize();
		if (e != cudaSuccess) {
			cerr << " Error: forward_avgpool failed!\n";
			return false;
		}
	}
	return true;
}