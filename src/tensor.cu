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
		int g = GPUGridSize();
		int b = GPUBlockSize();
		int threads = g * b;
		tensor_set_kernel<<<g, b>>>(gpu_data, (int)elements, threads, val);
		cudaError_t err = cudaDeviceSynchronize();
		if (cudaSuccess != err) {
			cerr << "Error: FloatTensor4D.operator= returned " << err << endl;
			fine = false;
		}
	}
	fine = true;
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
		int g = GPUGridSize();
		int b = GPUBlockSize();
		int threads = g * b;
		tensor_add_kernel <<<g, b >>> (gpu_data, right.gpu_data, (int)elements, threads);
	}
	else if (channels == right.channels) 	{
		int g = GPUGridSize(batch);
		int b = GPUBlockSize(channels);
		tensor_add_kernel_ex <<<g, b >>> (gpu_data, right.gpu_data, batch, channels ,elements_2d);
	}		 
	else {
		cerr << "Not compatible!\n";
		return false;
	}
	cudaError_t err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		cerr << "Error: FloatTensor4D.Add returned " << err << endl;
		return false;
	}
	return true;
}
bool add_in_gpu(float* dest, const float* src, int elements) {
	if (NULL == dest || NULL == src || 0 == elements) {
		return false;
	}
	int g = GPUGridSize();
	int b = GPUBlockSize();
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
 
	int g = GPUGridSize();
	int b = GPUBlockSize();
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
	int g = GPUGridSize();
	int b = GPUBlockSize();
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
	if (0.0f == val) {
		return cudaMemset(gpu_data, 0, bytes) == cudaSuccess;
	}
	int g = GPUGridSize();
	int b = GPUBlockSize();
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
	int g = GPUGridSize();
	int b = GPUBlockSize();
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
	int g = GPUGridSize();
	int b = GPUBlockSize();
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
	int g = GPUGridSize();
	int b = GPUBlockSize();
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
	int stride_w;
	int stride_h ;
	bool  is_nchw;

};
__global__ static void tensor_upsample_kernel(sample_params p) {  
	float* src = p.src_mem;
	float* dst = p.dst_mem;
	for (int b = 0; b < p.batchs; b++) {
		if (p.is_nchw) {
			for (int c = 0; c < p.channels; c++, src += p.src_e2d, dst += p.dst_e2d) {
				for (int y = blockIdx.x; y < p.src_height; y += gridDim.x) { 
					int dst_y = y * p.stride_h;
					int src_off = y * p.src_width;
					for (int x = threadIdx.x; x < p.src_width; x += blockDim.x) { 
						int dst_x = x * p.stride_w;
						int src_index = src_off  + x;
						for (int i = 0; i < p.stride_h; i++) {
							int dst_off = (dst_y + i) * p.dst_width + dst_x;
							for (int j = 0; j < p.stride_w; j++) { 
								dst[dst_off + j] = src[src_index];
							}
						}
						
					}
				}
			}
		}
		else { // nchw
			src += p.src_e3d;
			dst += p.dst_e3d;
			for (int y = blockIdx.x; y < p.src_height; y += gridDim.x) {
				int dst_y = y * p.stride_h;
				int src_off = y * p.src_width;
				for (int x = threadIdx.x; x < p.src_width; x += blockDim.x) {
					int dst_x = x * p.stride_w;
					int src_index = (src_off + x) * p.channels ;
					for (int i = 0; i < p.stride_h; i++) {
						int dst_off = (dst_y + i) * p.dst_width;
						for (int j = 0; j < p.stride_w; j++) {
							int dst_index = (dst_off + dst_x + j) * p.channels ;
							for (int c = 0; c < p.channels; c++) {
								dst[dst_index + c] = src[src_index + c];
							}
						}
					}

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
		batch, channels,stride_w,stride_h, order == TO_NCHW };
 
	tensor_upsample_kernel<<<g,b>>>(params); 
	cudaError_t err = cudaDeviceSynchronize();

	if (err != cudaSuccess) {
		cerr << "FloatTensor4D.UpSample failed - cudaSynchronize failed err " << err << "!\n";
		return false;
	} 
	return true;
}
__global__ static void tensor_downsample_kernel(sample_params p) {
	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	
	int total_elements = p.batchs * p.dst_e3d;
	int threads = gridDim.x * blockDim.x;

	while (thread_index < total_elements) {
		
		int b = thread_index / p.dst_e3d;
		int temp = thread_index % p.dst_e3d;
		int c, x , y ; 
		if (p.is_nchw) {
			c = temp / p.dst_e2d; 
			temp %= p.dst_e2d;
			y = temp / p.dst_width;
			x = temp % p.dst_width; 
			 
			float* src = p.src_mem + b * p.src_e3d + c * p.src_e2d;
			int src_y = y * p.stride_h;
			for (int i = 0; i < p.stride_h; i++, src_y++) {
				int src_x = x * p.stride_w;
				for (int j = 0; j < p.stride_w; j++, src_x++) {
					int src_index = src_y * p.src_width + src_x;					
					p.dst_mem[thread_index] += src[src_index];
				}
			}
		}
		else {
			c = temp % p.channels;
			temp /= p.channels;
			y = temp / p.dst_width;
			x = temp % p.dst_width;
			float* src = p.src_mem + b * p.src_e3d ; 
			for (int i = 0; i < p.stride_h; i++) {
				for (int j = 0; j < p.stride_w; j++) {
					int src_index = (y * p.stride_h + i) * p.src_width + (x * p.stride_w) + j;
					p.dst_mem[i] += src[src_index * p.channels + c];
				}
			}
		}
		thread_index += threads;
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
	
	int g = GPUGridSize();
	int b = GPUBlockSize();  
	sample_params params = {
		gpu_data , width, height, elements_2d, elements_3d,
		result.gpu_data, result.width, result.height, result.elements_2d, result.elements_3d,
		batch, channels,stride_w,stride_h, order == TO_NCHW};
	
	tensor_downsample_kernel <<<g, b>>>(params);
	 
	
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
	int g = GPUGridSize();
	int b = GPUBlockSize();
	tensor_random_kernel <<<g, b>>>(gpu_data, elements, g * b, GetTickCount(), min_, max_);
	cudaError_t err = cudaDeviceSynchronize();
	return (err == cudaSuccess);
}
 
__global__ void backward_bias_kernel(float *bias_updates, float *delta, int batch, int channels, int size) {
	for (int c = blockIdx.x; c < channels; c += gridDim.x) {
		for (int b = threadIdx.x; b < batch; b += blockDim.x) {			
			int channel_index = b * channels + c;			
			int data_offset = channel_index * size; 
			float* data = delta + data_offset;
			for (int i = 0; i < size; i++) {
				bias_updates[c] += data[i];
			}
		}
	}
}
bool backward_bias_gpu(float *bias_updates, float *delta, int batch, int channels, int size) {
	int g = GPUGridSize();
	int b = GPUBlockSize();
	backward_bias_kernel <<<g,b>>>(bias_updates, delta, batch, channels, size);
	cudaError_t err = cudaDeviceSynchronize();
	return err == cudaSuccess;
}
 
/*
#define BLOCK 512
__global__ void backward_bias_kernel(float *bias_updates, float *delta, int batch, int n, int size)
{
	__shared__ float part[BLOCK];
	int i, b;
	int filter = blockIdx.x;
	int p = threadIdx.x;
	float sum = 0;
	for (b = 0; b < batch; ++b) {
		for (i = 0; i < size; i += BLOCK) {
			int index = p + i + size*(filter + n*b);
			sum += (p + i < size) ? delta[index] : 0;
		}
	}
	part[p] = sum;
	__syncthreads();
	if (p == 0) {
		for (i = 0; i < BLOCK; ++i) bias_updates[filter] += part[i];
	}
}
bool backward_bias_gpu(float *bias_updates, float *delta, int batch, int channels, int size)
{
	backward_bias_kernel <<<channels, BLOCK >>>(bias_updates, delta, batch, channels, size);
	cudaError_t err = cudaDeviceSynchronize();
	return err == cudaSuccess;
}
*/
__global__ static void f32_to_f16_kernel(__half* dst, const float* src, size_t n) {
	int threads = gridDim.x * blockDim.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	while (index < n) {
		dst[index] = __float2half(src[index]);
		index += threads;
	}
}
bool f32_to_f16(__half* dst, const float* src, size_t n) {
	int g = GPUGridSize();
	int b = GPUBlockSize();
	f32_to_f16_kernel<<<g, b>>>(dst, src, n);
	cudaError_t err = cudaDeviceSynchronize();
	return err == cudaSuccess;
}
__global__ static void f16_to_f32_kernel(float* dst, const __half* src, size_t n) {
	int threads = gridDim.x * blockDim.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	while (index < n) {
		dst[index] = __half2float(src[index]);
		index += threads;
	}
}
bool f16_to_f32(float* dst, const __half* src, size_t n) {
	int g = GPUGridSize();
	int b = GPUBlockSize();
	f16_to_f32_kernel<<<g, b>>>(dst, src, n);
	cudaError_t err = cudaDeviceSynchronize();
	return err == cudaSuccess;
}
__global__ static void calc_weights_for_ir_kernel(float* w, const float* factors, int c_in, int size, int elements) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int threads = gridDim.x * blockDim.x; 
	while (index < elements) { 
		int c = (index / size) % c_in; 
		w[index] *= factors[c];		
		index += threads;
	}
}
bool calc_weights_for_ir(float* w, const float* factors, int c_in, int size, int elements) {
 
	int g = GPUGridSize();
	int b = GPUBlockSize();
	calc_weights_for_ir_kernel<<<g,b>>>(w,factors, c_in, size, elements);
	cudaError_t err = cudaDeviceSynchronize();

	return err == cudaSuccess;
}