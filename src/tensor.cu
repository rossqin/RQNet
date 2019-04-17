#include "stdafx.h"
#include "cuda_tensor.h"
#include <cuda_fp16.h>

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

__global__ void tensor_upsample_kernel(void* dst_mem, int width, int height , void* src_mem,
	int batch, int channels, int stride_w, int stride_h, cudnnDataType_t data_type, cudnnTensorFormat_t data_format){
 
	int threads = gridDim.x * blockDim.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int c_size_dest = channels * width * height;
	int elements = batch * c_size_dest;

	int src_width = width / stride_w;
	int src_height = height / stride_h;
	int c_size_src = channels * src_width * src_height;

	while (index < elements) {
		int b = index / c_size_dest;
		int temp = index % c_size_dest;
		int c, h, w, index_src;
		if (data_format == CUDNN_TENSOR_NCHW) {
			c = temp / (width * height);
			temp = temp % (width * height);
			h = temp / width;
			w = temp % width;
			index_src = b * c_size_src + c * (src_width * src_height) +
				(h / stride_h) * src_width + w / stride_w;
		}
		else {
			h = temp / (width * channels);
			temp = temp % (width * channels);
			w = temp / channels;
			c = temp % channels;
			index_src = b * c_size_src + (h / stride_h) * (src_width * channels) +
				(w / stride_w) * channels + c;
		}
		if (data_type == CUDNN_DATA_FLOAT) {
			float* src = reinterpret_cast<float*>(src_mem);
			float* dst = reinterpret_cast<float*>(dst_mem);
			dst[index] = src[index_src];
		}
		else {
			__half* src = reinterpret_cast<__half*>(src_mem);
			__half* dst = reinterpret_cast<__half*>(dst_mem);
			dst[index] = src[index_src];
		}
		index += threads;
	} 
}
__global__ void tensor_downsample_kernel(void* dst_mem, int width, int height, void* src_mem,
	int batch, int channels, int stride_w, int stride_h, cudnnDataType_t data_type, cudnnTensorFormat_t data_format) {

	int threads = gridDim.x * blockDim.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int c_size_dest = channels * width * height;
	int elements = batch * c_size_dest;

	int src_width = width * stride_w;
	int src_height = height * stride_h;
	int c_size_src = channels * src_width * src_height;

	while (index < elements) {
		int b = index / c_size_dest;
		int temp = index % c_size_dest;
		int c, h, w,  src_h, src_w, index_src;
		if (data_format == CUDNN_TENSOR_NCHW) {
			c = temp / (width * height);
			temp = temp % (width * height);
			h = temp / width;
			w = temp % width; 
		}
		else {
			h = temp / (width * channels);
			temp = temp % (width * channels);
			w = temp / channels;
			c = temp % channels; 
		}
		src_h = h * stride_h;
		
		
		if (data_type == CUDNN_DATA_FLOAT) {
			float* src = reinterpret_cast<float*>(src_mem);
			float* dst = reinterpret_cast<float*>(dst_mem);
			dst[index] = 0.0f; 
			for (int i = 0; i < stride_h; i++, src_h++) {
				src_w = w * stride_w;
				for (int j = 0; j < stride_w; j++, src_w++) { 
					if (data_format == CUDNN_TENSOR_NCHW) {
						index_src = b * c_size_src + c * (src_width * src_height) +
							src_h * src_width + src_w;
					}
					else {
						index_src = b * c_size_src + src_h * (src_width * channels) +
							src_w * channels + c;
					}
					dst[index] += src[index_src];
				}
			}
		}
		else {
			__half* src = reinterpret_cast<__half*>(src_mem);
			__half* dst = reinterpret_cast<__half*>(dst_mem);
			dst[index] = __float2half(0.0f);
			for (int i = 0; i < stride_h; i++, src_h++) {
				src_w = w * stride_w;
				for (int j = 0; j < stride_w; j++, src_w++) {
					if (data_format == CUDNN_TENSOR_NCHW) {
						index_src = b * c_size_src + c * (src_width * src_height) +
							src_h * src_width + src_w;
					}
					else {
						index_src = b * c_size_src + src_h * (src_width * channels) +
							src_w * channels + c;
					} 
					dst[index] = __hadd(dst[index], src[index_src]);
				}
			}
		}
		index += threads;
	}
}

bool CudaTensor::UpSample(CudaTensor& output, int stride_w, int stride_h) {
	if (stride_w <= 0 || stride_w <= 0 || 0 == elements) return false;

	int w_o = w * stride_w;
	int h_o = h * stride_h;

	if (output.n != n || output.c != c || output.w != w_o || output.h != h_o) {
		cerr << " Error: Wrong result demension in tensor upsample !\n";
		return false;
	}
	if (output.data_type != data_type || output.data_format != data_format) {
		cerr << " Error: Inconsistent data types in tensor upsample !\n";
		return false;
	}
	int g = GPUGridSize();
	int b = GPUBlockSize();
	switch (data_type) {
	case CUDNN_DATA_FLOAT:
	case CUDNN_DATA_HALF:
		tensor_upsample_kernel<<<g,b>>>(output.gpu_data, output.w, output.h, gpu_data, n, c, stride_w, stride_h, data_type, data_format);
		break;
	default:
		cerr << " Error: Only support FP16 or FP32!\n";
		return false;
	}  
	cudaError_t err = cudaDeviceSynchronize();

	if (err != cudaSuccess) {
		cerr << " Error: CudaTensor.UpSample failed - cudaSynchronize failed err " << err << "!\n";
		return false;
	}
	return true;
}
bool CudaTensor::DownSample(CudaTensor& output, int stride_w, int stride_h) {
	if (stride_w <= 0 || stride_w <= 0 || 0 == elements) return false;

	int w_o = w / stride_w;
	int h_o = h / stride_h;

	if (output.n != n || output.c != c || output.w != w_o || output.h != h_o) {
		cerr << " Error: Wrong result demension in tensor upsample !\n";
		return false;
	}
	if (output.data_type != data_type || output.data_format != data_format) {
		cerr << " Error: Inconsistent data types in tensor upsample !\n";
		return false;
	}
	int g = GPUGridSize();
	int b = GPUBlockSize(); 

	switch (data_type) {
	case CUDNN_DATA_FLOAT:
	case CUDNN_DATA_HALF:
		tensor_downsample_kernel <<<g, b>>>(output.gpu_data, output.w, output.h, gpu_data, n, c, stride_w, stride_h, data_type, data_format);
		break;
	default:
		cerr << " Error: Only support FP16 or FP32!\n";
		return false;
	}
	cudaError_t err = cudaDeviceSynchronize();

	if (err != cudaSuccess) {
		cerr << " Error: CudaTensor.DownSample failed - cudaSynchronize failed err " << err << "!\n";
		return false;
	}
	return true;
}
__global__ static void tensor_add_kernel(void* data, const void* op, int elements, cudnnDataType_t data_type) {
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	int threads = gridDim.x * blockDim.x;
	while (index < elements) {
		if (data_type == CUDNN_DATA_FLOAT) {
			float* dst = reinterpret_cast<float *>(data);
			const float* src = reinterpret_cast<const float *>(op);
			dst[index] += src[index];
		}
		else {
			__half* dst = reinterpret_cast<__half *>(data);
			const __half* src = reinterpret_cast<const __half *>(op);
			dst[index] = __hadd(dst[index], src[index]);
		}
		index += threads;
	}
}
__global__ static void tensor_add_kernel_ex(void* data, const void* op, int batch, int channels, int height, int width, int op_batch, cudnnDataType_t data_type, cudnnTensorFormat_t data_format) {
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	int threads = gridDim.x * blockDim.x;
	int c_size = channels * height * width;
	int elements = batch * c_size;
	while (index < elements) {
		int b = index / c_size;
		int temp = index % c_size;
		int c ;
		if (data_format == CUDNN_TENSOR_NCHW) {
			c = temp / (width * height);
			temp = temp % (width * height); 
		}
		else {
			//int h = temp / (width * channels);
			temp = temp % (width * channels); 
			c = temp % channels;
		}
		if (data_type == CUDNN_DATA_FLOAT) {
			float* dst = reinterpret_cast<float *>(data);
			const float* src = reinterpret_cast<const float *>(op);
			if(batch == op_batch )
				dst[index] += src[b * channels + c];
			else 
				dst[index] += src[c];
		}
		else {
			__half* dst = reinterpret_cast<__half *>(data);
			const __half* src = reinterpret_cast<const __half *>(op);
			if (batch == op_batch)
				dst[index] = __hadd(dst[index], src[b * channels + c]);
			else
				dst[index] = __hadd(dst[index], src[c]);
		}
		index += threads;
	}
}

bool CudaTensor::Add(const CudaTensor& op) {
	if (!op.gpu_data || !op.elements) {
		return true;
	}
	if (!gpu_data) {
		(*this) = op;
		return SameShape(op);
	}
	if (op.data_type != data_type || op.data_format != data_format) {
		cerr << " Error: Inconsistent data types in tensor add !\n";
		return false;
	}
	if(data_type != CUDNN_DATA_FLOAT && data_type != CUDNN_DATA_HALF){
		cerr << " Error: Unsportted data format in tensor add !\n";
		return false;
	}
	if (n != op.n && op.n != 1) {
		cerr << " Error: Inconsistent batches in tensor add !\n";
		return false;
	}
	if (elements == op.elements) {
		int g = GPUGridSize();
		int b = GPUBlockSize(); 
		tensor_add_kernel <<<g, b>>> (gpu_data, op.gpu_data, elements, data_type);
	}
	else if (c == op.c && (op.h == 1 && op.w == 1)) {
		int g = GPUGridSize();
		int b = GPUBlockSize();
		tensor_add_kernel_ex <<<g,b >>> (gpu_data, op.gpu_data, n, c, h, w, op.n, data_type, data_format);
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
__global__ static void tensor_add_kernel(void* data, float op, int elements, cudnnDataType_t data_type) {
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	int threads = gridDim.x * blockDim.x;

	while (index < elements) {
		if (data_type == CUDNN_DATA_FLOAT) {
			float* dst = reinterpret_cast<float *>(data); 
			dst[index] += op;
		}
		else {
			__half* dst = reinterpret_cast<__half *>(data);
			__half hop = __float2half(op);
			dst[index] = __hadd(dst[index], hop);
		}
		index += threads;
	}
}
bool CudaTensor::Add(float op) {	 
	if (!gpu_data) {
		(*this) = op;
		return true;
	}	 
	if (data_type != CUDNN_DATA_FLOAT && data_type != CUDNN_DATA_HALF) {
		cerr << " Error: Unsportted data format in tensor add !\n";
		return false;
	} 
	int g = GPUGridSize();
	int b = GPUBlockSize(); 
	tensor_add_kernel <<<g, b>>> (gpu_data, op, elements, data_type);
	cudaError_t err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		cerr << "Error: FloatTensor4D.Add returned " << err << endl;
		return false;
	}
	return true;
}

__global__ static void tensor_muladd_kernel(void* data, float op_m, float op_a, int elements, cudnnDataType_t data_type) {
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	int threads = gridDim.x * blockDim.x;

	while (index < elements) {
		if (data_type == CUDNN_DATA_FLOAT) {
			float* dst = reinterpret_cast<float *>(data);
			dst[index] = op_m * dst[index] + op_a;
		}
		else {
			__half* dst = reinterpret_cast<__half *>(data);
			__half hop = __float2half(op_m);
			dst[index] = __hmul(dst[index], hop);
			if (op_a != 0.0f) {
				hop = __float2half(op_a);
				dst[index] = __hadd(dst[index], hop);
			}
		}
		index += threads;
	}
}
bool CudaTensor::MulAdd(float op_m, float op_a) {
	if (!gpu_data) {
		return false;
	}
	if (data_type != CUDNN_DATA_FLOAT && data_type != CUDNN_DATA_HALF) {
		cerr << " Error: Unsportted data format in tensor add !\n";
		return false;
	}
	int g = GPUGridSize();
	int b = GPUBlockSize();
	tensor_muladd_kernel <<<g, b >>> (gpu_data, op_m, op_a, elements, data_type);
	cudaError_t err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		cerr << "Error: FloatTensor4D.Add returned " << err << endl;
		return false;
	}
	return true;
}

__global__ static void tensor_muladd_kernel_ex(void* data, const void* op_m, const void* op_a, int batch, int channels, int height, int width, cudnnDataType_t data_type, cudnnTensorFormat_t data_format) {
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	int threads = gridDim.x * blockDim.x;
	int c_size = channels * height * width;
	int elements = batch * c_size;
	while (index < elements) {
		int b = index / c_size;
		int temp = index % c_size;
		int c;
		if (data_format == CUDNN_TENSOR_NCHW) {
			c = temp / (width * height);
			temp = temp % (width * height);
		}
		else {
			//int h = temp / (width * channels);
			temp = temp % (width * channels);
			c = temp % channels;
		}
		if (data_type == CUDNN_DATA_FLOAT) {
			float* dst = reinterpret_cast<float *>(data);
			const float* src_m = reinterpret_cast<const float *>(op_m);
			const float* src_a = reinterpret_cast<const float *>(op_a);
			dst[index] = dst[index] * src_m[c] + src_a[c];
		}
		else {
			__half* dst = reinterpret_cast<__half *>(data);
			const __half* src_m = reinterpret_cast<const __half *>(op_m);
			const __half* src_a = reinterpret_cast<const __half *>(op_a);
			__half temp = __hmul(dst[index], src_m[c]);
			dst[index] = __hadd(temp, src_a[c]);
		}
		index += threads;
	}
}
bool CudaTensor::MulAdd(const CudaTensor& op_m, const CudaTensor& op_a) {
	if (!op_m.gpu_data || !op_m.elements || !op_a.gpu_data || !op_a.elements) {
		return true;
	}
	if (!gpu_data) {
		return false;
	}
	if (op_m.data_type != data_type || op_m.data_format != data_format
		||op_a.data_type != data_type || op_a.data_format != data_format) {
		cerr << " Error: Inconsistent data types in tensor multia !\n";
		return false;
	}
	if (data_type != CUDNN_DATA_FLOAT && data_type != CUDNN_DATA_HALF) {
		cerr << " Error: Unsportted data format in tensor add !\n";
		return false;
	}
	if (op_m.n != 1 || op_m.c != c || op_m.w != 1 || op_m.h != 1 ||
		op_a.n != 1 || op_a.c != c || op_a.w != 1 || op_a.h != 1) {
		cerr << " Error: Dims of operators must be [1x"<<c<<"x1x1]!\n";
		return false;
	}
	int g = GPUGridSize();
	int b = GPUBlockSize();
	tensor_muladd_kernel_ex <<<g, b >>> (gpu_data, op_m.gpu_data, op_a.gpu_data, n, c, h, w,  data_type, data_format);
 
	cudaError_t err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		cerr << "Error: FloatTensor4D.Add returned " << err << endl;
		return false;
	}
	return true;
}
__global__ static void sgd_update_kernel(void* params, void* updates, int elements, cudnnDataType_t data_type, float lr, float decay, float momentum) {
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	int threads = gridDim.x * blockDim.x;
	while (index < elements) {
		if (data_type == CUDNN_DATA_FLOAT) {
			float* dst = reinterpret_cast<float*>(params);
			float* src = reinterpret_cast<float*>(updates);
			src[index] -= (dst[index] * decay);
			dst[index] += (lr * src[index]);
			src[index] *= momentum;
		}
		else {
			__half* dst = reinterpret_cast<__half*>(params);
			__half* src = reinterpret_cast<__half*>(updates);
			__half temp = __hmul(dst[index], __float2half(decay));
			src[index] = __hsub(src[index],temp);
			temp = __hmul(src[index], __float2half(lr));
			dst[index] = __hsub(dst[index], temp);
			src[index] = __hmul(src[index] , __float2half(momentum)); 
		}
		index += threads;
	}
}
bool sgd_update(void* params, void* updates, int elements, cudnnDataType_t data_type, float lr, float decay, float momentum) {
	int g = GPUGridSize();
	int b = GPUBlockSize();
	sgd_update_kernel<<<g, b >>>(params, updates, elements, data_type, lr, decay, momentum);

	cudaError_t err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		cerr << "Error: FloatTensor4D.Add returned " << err << endl;
		return false;
	}
	return true;
}
/*
void* beta = params.BatchData(0);
void* gamma = params.BatchData(1);
void* running_mu = params.BatchData(2);
void* running_var = params.BatchData(3);
*/
__global__ static void fuse_batchnorm_kernel(void* filters, void* bias, void* batchnorm_params, int channels, int w, int h, int channels_in, cudnnDataType_t data_type) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int size = w * h;
	int elements = channels * size;
 
	while (index < channels) { 
		if (data_type == CUDNN_DATA_FLOAT) {
			float* filters_f = reinterpret_cast<float*>(filters);
			float* bias_f = reinterpret_cast<float*>(bias);
			float* beta = reinterpret_cast<float*>(batchnorm_params);
			float* gamma = beta + channels;
			float* mu = gamma + channels;
			float* var = mu + channels;
			float temp = gamma[index] / sqrt(var[index] * var[index] + 1.0e-5);
			float alpha = temp / size;
			bias_f[index] += (beta[index] - mu[index] * temp);
			for (int i = 0; i < channels_in; i++) {
				for (int j = 0; j < h; j++) {
					for (int k = 0; k < w; k++) {
						filters_f[i * elements + index * size + j * w + k] *= alpha;
					}
				}
			} 
		}
		else {
			__half* filters_h = reinterpret_cast<__half*>(filters);
			__half* bias_h = reinterpret_cast<__half*>(bias);
			__half* beta = reinterpret_cast<__half*>(batchnorm_params);
			__half* gamma = beta + channels;
			__half* mu = gamma + channels;
			__half* var = mu + channels;

			__half temp = __hdiv(gamma[index], hsqrt( __hadd(__hmul(var[index], var[index]), __float2half(1.0e-5))));
		 
			__half alpha = __hdiv(temp , __float2half(size));
			bias_h[index] = __hadd(bias_h[index],(beta[index] - mu[index] * temp));
			for (int i = 0; i < channels_in; i++) {
				for (int j = 0; j < h; j++) {
					for (int k = 0; k < w; k++) {
						int dest_index = i * elements + index * size + j * w + k;
						filters_h[dest_index] = __hmul(filters_h[dest_index], alpha);
					}
				}
			}
		}

		index += blockDim.x * gridDim.x;
	}


}
bool fuse_batchnorm(void* filters, void* bias, void* batchnorm_params, int channels, int w, int h, int channels_in, cudnnDataType_t data_type) {
	int g = GPUGridSize();
	int b = GPUBlockSize();
	fuse_batchnorm_kernel<<<g, b >>>(filters, bias, batchnorm_params, channels, w, h, channels_in, data_type);

	cudaError_t err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		cerr << " Error: fuse_batchnorm returned " << err << endl;
		return false;
	}
	return true;
}
__global__ static void one_stride_pooling_patch_kernel(void* out, void* in, int batch, int channels, int width, int height, cudnnDataType_t data_type, cudnnTensorFormat_t data_format, bool forwarding) {
	int index = blockIdx.x * blockDim.x + threadIdx.x; 
	int threads = gridDim.x * blockDim.x;
	int size = height * width;
	int c_size = channels * size;
	int elements = batch * c_size;
	while (index < elements) {
		int b = index / c_size;
		int temp = index % c_size;
		int c, w, h, index1;
		if (data_format == CUDNN_TENSOR_NCHW) {
			c = temp / size;
			temp = temp % size;
			h = temp / width;
			w = temp % width; 
			//index1 = b * c_size + c * size + (h + 1) * (width + 1) + width + 1;
			index1 = b * c_size + c * size + h * (width + 1) + width;
		}
		else {
			size = width * channels;
			h = temp / size;
			temp = temp % size;
			w = temp / channels;
			c = temp % channels;
			//index1 = b * c_size +  (h + 1) * (width + 1) * channels + (width + 1) * channels + c;
			index1 = b * c_size + h * (width + 1) * channels + width * channels + c;
		}
		if (data_type == CUDNN_DATA_FLOAT) {
			float* fout = reinterpret_cast<float*>(out);
			float* fin = reinterpret_cast<float*>(in);
			if(forwarding)
				fout[index] = fin[index1];
			else
				fout[index1] = fin[index];
		}
		else {
			__half* hout = reinterpret_cast<__half*>(out);
			__half* hin = reinterpret_cast<__half*>(in);
			if (forwarding)
				hout[index] = hin[index1];
			else
				hout[index1] = hin[index];
		}
		index += threads;
	}
}
bool one_stride_pooling_patch(CudaTensor& out, const CudaTensor& in, bool forwarding) {
	int g = GPUGridSize();
	int b = GPUBlockSize();
	int h, w;
	if (forwarding) {
		h = out.Height();
		w = out.Width();
	}
	else {
		h = in.Height();
		w = in.Width();
	}
	one_stride_pooling_patch_kernel<<<g, b >>>(out, in,out.Batch(),out.Channel(),w,h ,
		out.DataType(),out.DataFormat(), forwarding);

	cudaError_t err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		cerr << " Error: one_stride_pooling_patch returned " << err << endl;
		return false;
	}
	return true;
}
