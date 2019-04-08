#include "stdafx.h"
#include "tensor.h"


__global__ static void tensor_set_kernel(float* data, int elements, int threads, float val) {
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	while (index < elements) {
		data[index] = val;
		index += threads;
	}
}
const FloatTensor4D& FloatTensor4D::operator=(float val) {
	if (NULL == gpu_data) return *this;
	if (0.0 == val) {
		fine = (cudaSuccess == cudaMemset(gpu_data, 0, bytes));
	}
	else {
		int g = GPUGridSize(9999);
		int b = GPUBlockSize(9999);
		int threads = g * b;
		tensor_set_kernel<<<g, b>>>(gpu_data, (int)elements, threads, val);
		cudaError_t err = cudaDeviceSynchronize();
		if (cudaSuccess != err) {
			cerr << "Error: FloatTensor4D.operator= returned " << err << endl;
			fine = false;
		}
	}
	return *this;
}
__global__ static void tensor_add_kernel(float* data, const float* op, int elements, int threads) {
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	while (index < elements) {
		data[index] += op[index];
		index += threads;
	}
}
__global__ static void tensor_add_kernel_ex(float* data, const float* op, int batch, int channels, int elements ) {
	
	int element3d = elements * channels;
	
	for (int b = blockIdx.x; b < batch; b += gridDim.x) {
		float* batch_data = data + b * element3d;
		for (int c = threadIdx.x; c < channels; c += blockDim.x) {
			float* channel_data = batch_data + c * elements;
			for (int i = 0; i < elements; i++) {
				channel_data[i] += op[c];
			}
		}
	}
	 
	
}
bool FloatTensor4D::Add(const FloatTensor4D& right) {
	if (NULL == right.gpu_data || 0 == right.elements) {
		return true;
	}
	if (NULL == gpu_data) {
		this->operator=(right);
		return fine;
	} 
	
	if (elements == right.elements) {
		int g = GPUGridSize(9999);
		int b = GPUBlockSize(9999);
		int threads = g * b;
		tensor_add_kernel <<<g, b >>> (gpu_data, right.gpu_data, (int)elements, threads);
	}
	else if (channels == right.channels) 	{
		int g = GPUGridSize(batch);
		int b = GPUBlockSize(channels);
		tensor_add_kernel_ex <<<g, b >>> (gpu_data, right.gpu_data, batch, channels ,elements_2d);
	}		 
	else
		return false;
	cudaError_t err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		cerr << "Error: FloatTensor4D.Add returned " << err << endl;
		return false;
	}
	return true;
}
bool add_in_gpu(float* dest, float* src, int elements) {
	if (NULL == dest || NULL == src || 0 == elements) {
		return false;
	}
	int g = GPUGridSize(9999);
	int b = GPUBlockSize(9999);
	int threads = g * b;
	tensor_add_kernel <<<g, b>>>(dest, src, (int)elements, threads);
	cudaError_t err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		cerr << "Error: add_in_gpu returned " << err << endl;
		return false;
	}
	return true;
}
bool FloatTensor4D::Add(const float * vals, size_t length) {
	if (NULL == vals || length != elements) {
		return false;
	}
	if (NULL == gpu_data) {		 
		return false;
	}
 
	int g = GPUGridSize(9999);
	int b = GPUBlockSize(9999);
	int threads = g * b;
	tensor_add_kernel <<<g, b>>>(gpu_data, vals, (int)elements, threads);
	cudaError_t err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		cerr << "Error: FloatTensor4D.Add returned " << err << endl;
		return false;
	}
	return true; 
}
__global__ static void tensor_add_kernel(float* data, int elements, int threads, float val) {
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	while (index < elements) {
		data[index] += val;
		index += threads;
	}
}
bool FloatTensor4D::Add(float val) {
	int g = GPUGridSize(9999);
	int b = GPUBlockSize(9999);
	int threads = g * b;
	tensor_add_kernel <<<g, b >>>(gpu_data, (int)elements, threads, val);
	cudaError_t err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		cerr << "Error: FloatTensor4D.Add returned " << err << endl;
		return false;
	}
	return true;
}
__global__ static void tensor_mul_kernel(float* data, int elements, int threads, float val) {
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	while (index < elements) {
		data[index] += val;
		index += threads;
	}

}
bool FloatTensor4D::Mul(float val) {
	if (NULL == gpu_data) return false;
	if (1.0f == val) return true;
	int g = GPUGridSize(9999);
	int b = GPUBlockSize(9999);
	int threads = g * b;
	tensor_mul_kernel <<<g, b >>>(gpu_data, (int)elements, threads, val);
	cudaError_t err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		cerr << "Error: FloatTensor4D.Mul returned " << err << endl;
		return false;
	}
	return true;
}
__global__ static void tensor_muladd_kernel(float* data, int elements, int threads, float scale, float bias) {
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	while (index < elements) {
		data[index] = scale * data[index] + bias;
		index += threads;
	}
}
bool FloatTensor4D::MulAdd(float scale, float bias) {
	if (NULL == gpu_data) return false;
	if (1.0f == scale) return Add(bias);
	int g = GPUGridSize(9999);
	int b = GPUBlockSize(9999);
	int threads = g * b;
	tensor_muladd_kernel <<<g, b >>>(gpu_data, (int)elements, threads, scale, bias);
	cudaError_t err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		cerr << "Error: FloatTensor4D.MulAdd returned " << err << endl;
		return false;
	}
	return true;
}
__global__ static void tensor_muladd_kernel(float* data, float* op, int elements, int threads, float scale) {
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	while (index < elements) {
		data[index] = scale * data[index] + op[index];
		index += threads;
	}
}
bool FloatTensor4D::MulAdd(float scale, const FloatTensor4D& right) {
	if (NULL == gpu_data || elements != right.elements) return false;
	if (1.0f == scale) return Add(right);
	int g = GPUGridSize(9999);
	int b = GPUBlockSize(9999);
	int threads = g * b;
	tensor_muladd_kernel <<<g, b >>>(gpu_data, right.gpu_data, (int)elements, threads, scale);
	cudaError_t err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		cerr << "Error: FloatTensor4D.MulAdd returned " << err << endl;
		return false;
	}
	return true;
}
__global__ static void tensor_addscale_kernel(float* data, float* op, int elements, int threads, float scale) {
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	while (index < elements) {
		data[index] += scale * op[index];
		index += threads;
	}
}
bool FloatTensor4D::AddScale(const FloatTensor4D& right, float scale) {
	if (NULL == gpu_data || elements != right.elements) return false;
	if (1.0f == scale) return Add(right);
	int g = GPUGridSize(9999);
	int b = GPUBlockSize(9999);
	int threads = g * b;
	tensor_addscale_kernel <<<g, b >>>(gpu_data, right.gpu_data, (int)elements, threads, scale);
	cudaError_t err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		cerr << "Error: FloatTensor4D.MulAdd returned " << err << endl;
		return false;
	}
	return true;
} 
struct sample_params {
	float* src_mem;
	int src_width;
	int src_height;
	int src_e2d;
	int src_e3d;

	float* dst_mem;
	int dst_width;
	int dst_height;
	int dst_e2d;
	int dst_e3d;

	int batchs;
	int channels;
	float stride_w;
	float stride_h ; 
	float stride_w_reci;
	float stride_h_reci; 
	

};
__global__ static void tensor_upsample_kernel_nchw(sample_params params) {  
	float* src = params.src_mem;
	float* dst = params.dst_mem;
	for (int b = 0; b < params.batchs; b++) {
		for (int c = 0; c < params.channels; c++, src += params.src_e2d , dst += params.dst_e2d) {
			for (int y = blockIdx.x; y < params.dst_height; y += gridDim.x) {
				int src_y = (int)(y * params.stride_h_reci);
				for (int x = threadIdx.x; x < params.dst_width; x += blockDim.x) {
					int src_x = (int)(x * params.stride_w_reci);
					dst[ y * params.dst_width + x] = src[ src_y * params.src_width + src_x];
				}
			}
		}
	}	
}
__global__ static void tensor_upsample_kernel_nhwc(sample_params params) {
	int x = threadIdx.x;
	float* src = params.src_mem;
	float* dst = params.dst_mem;
	for (int b = 0; b < params.batchs; b++, src += params.src_e3d,
		dst += params.dst_e3d) {		
		for (int y = blockIdx.x; y < params.dst_height; y += gridDim.x) {
			int src_y = (int)(y * params.stride_h_reci);
			for (int x = threadIdx.x; x < params.dst_width; x += blockDim.x) {
				int src_x = (int)(x * params.stride_w_reci);
				int dst_p = ( y * params.dst_width + x) * params.channels;
				int src_p = (src_y * params.src_width + src_x) * params.channels;
				for (int c = 0; c < params.channels; c++) {					
					dst[dst_p + c] = src[src_p + c];
				}
			}
		}
	}
}

bool FloatTensor4D::UpSample(FloatTensor4D& result, int stride_w, int stride_h )const {
	if (stride_w <= 0 || stride_w <= 0 || 0 == elements) return false;

	int r_width = width * stride_w;
	int r_height = height * stride_h;

	if (result.batch != batch || result.channels != channels ||
		result.width != r_width || result.height != r_height ||
		result.order != order) {
		cerr << "Error: Wrong result demension in tensor upsample !\n";
		return false;
	}

	int g = GPUGridSize(height);
	int b = GPUBlockSize(width);

	sample_params params = {
		gpu_data , width, height, elements_2d, elements_3d,
		result.gpu_data, result.width, result.height, result.elements_2d, result.elements_3d,
		batch, channels,stride_w,stride_h, 1.0 / stride_w,1.0 / stride_h  };
	if(order == TO_NCHW)
		tensor_upsample_kernel_nchw<<<g,b>>>(params);
	else 
		tensor_upsample_kernel_nhwc<<<g,b>>>(params);
	cudaError_t err = cudaDeviceSynchronize();

	if (err != cudaSuccess) {
		cerr << "FloatTensor4D.UpSample failed - cudaSynchronize failed err " << err << "!\n" ;
		cudaGetLastError(); 

		return false;
	} 
	return true;
}
__global__ static void tensor_downsample_kernel_nchw(sample_params params) {
	float* src = params.src_mem;
	float* dst = params.dst_mem;
	for (int b = 0; b < params.batchs; b++) {
		for (int c = 0; c < params.channels; c++, src += params.src_e2d, dst += params.dst_e2d) {
			for (int y = blockIdx.x; y < params.dst_height; y += gridDim.x) {
				int src_y = y * params.stride_h ;
				for (int x = threadIdx.x; x < params.dst_width; x += blockDim.x) {
					int src_x =  x * params.stride_w ;
					int index = y * params.dst_width + x;
					dst[index] = 0.0;
					for (int i = 0; i < params.stride_h; i++) {
						for (int j = 0; j < params.stride_w; j++) {
							dst[index] += src[(src_y + i) * params.src_width + src_x + j];
						}
					}
				}
			}
		}
	}
	 
}
bool FloatTensor4D::DownSample(FloatTensor4D& result, int stride_w, int stride_h )const {
	if (stride_w <= 0 || stride_w <= 0 || 0 == elements) return false;

	int r_width = width / stride_w;
	int r_height = height / stride_h;

	if (result.batch != batch || result.channels != channels ||
		result.width != r_width || result.height != r_height ||
		result.order != order) {
		cerr << "Error: Wrong result demension in tensor upsample !\n";
		return false;
	}
	
	int g = GPUGridSize(r_height);
	int b = GPUBlockSize(r_width);

	sample_params params = {
		gpu_data , width, height, elements_2d, elements_3d,
		result.gpu_data, result.width, result.height, result.elements_2d, result.elements_3d,
		batch, channels,stride_w,stride_h, 1.0, 1.0 };
	if (order == TO_NCHW)
		tensor_downsample_kernel_nchw <<<g, b>>>(params);
	else {
		//TODO: finish downsample under NHWC case
		return false;
	}
	
	cudaError_t err = cudaDeviceSynchronize();

	if (err != cudaSuccess) {
		cerr << "FloatTensor4D.DownSample failed - cudaSynchronize failed err " << err << "!\n";
		cudaGetLastError();

		return false;
	}
	
	return true;
}

__global__ static void tensor_random_kernel(float* data, int elements, int threads, size_t seed, float min_, float max_) {
	curandState s;
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	curand_init(seed, 0, index, &s);
	while (index < elements) {
		data[index] = curand_uniform(&s);
		if (min_ != 0.0f || max_ != 1.0f) {
			data[index] = data[index] * (max_ - min_) + min_;
		}
		index += threads;
	}
}
bool FloatTensor4D::Randomize(float min_, float max_) {
	if (0 == elements) return false;
	int g = GPUGridSize(9999);
	int b = GPUBlockSize(9999);
	tensor_random_kernel <<<g, b >>>(gpu_data, elements, g * b, GetTickCount(), min_, max_);
	cudaError_t err = cudaDeviceSynchronize();
	return (err == cudaSuccess);
} 