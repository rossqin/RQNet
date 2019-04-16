#include "stdafx.h"
#include "cuda_tensor.h"
#include "network.h"
#include <memory>

CudaTensor::CudaTensor(const CudaTensor & right) {
	data_format = right.data_format;
	data_type = right.data_type;
	byte_per_element = right.byte_per_element;
	n = right.n;
	c = right.c;
	h = right.h;
	w = right.w; 
	gpu_data = nullptr;
	elements = right.elements;
	bytes = right.bytes;
	desc = nullptr;
	cudnnCreateTensorDescriptor(&desc);
	cudaMalloc(&gpu_data, bytes);
	cudaMemcpy(gpu_data, right.gpu_data, bytes, cudaMemcpyDeviceToDevice);
	bad_flag = false;
}

CudaTensor::CudaTensor(cudnnDataType_t t, cudnnTensorFormat_t f) {
	n = 0;
	c = 0;
	h = 0;
	w = 0;
	data_format = f;
	data_type = t;
	gpu_data = nullptr;
	elements = 0;
	bytes = 0;
	desc = nullptr;
	bad_flag = false;
	switch (data_type)
	{
	case CUDNN_DATA_FLOAT:
	case CUDNN_DATA_INT32:
		byte_per_element = 4;
		break;
	case CUDNN_DATA_HALF:
		byte_per_element = 2;
		break;
	default:
		byte_per_element = 1;
		break;
	}
	 
}
CudaTensor::~CudaTensor() {
	if (desc) cudnnDestroyTensorDescriptor(desc);
	if (gpu_data) cudaFree(gpu_data);
}
char* CudaTensor::BatchData(int b) const {
	if(!gpu_data || b < 0 || b>= n) return nullptr; 
	return reinterpret_cast<char*>(gpu_data) + Elements3D() * byte_per_element;
}
bool CudaTensor::Init(int n_, int c_, int h_, int w_) {
	
	if (n_ == n && c_ == c && h_ == h && w_ == w) {
		if (bytes != elements * byte_per_element) {
			bytes = elements * byte_per_element;
			cudaFree(gpu_data);
			cudaMalloc(&gpu_data, bytes);
		}
		return cudaSuccess == cudaMemset(gpu_data, 0, bytes);
	}
	if (gpu_data) { 
		cudaError_t err = cudaFree(gpu_data);
		gpu_data = NULL;
		if (err != cudaSuccess) {
			cerr << "cudaFree at 0x" << hex << setw(10) << setfill('0') << (size_t)gpu_data << " return " << (int)err << endl;
		}
	}
	if (nullptr == desc) {
		if(CUDNN_STATUS_SUCCESS != cudnnCreateTensorDescriptor(&desc))
			return false;
	}
	n = n_;
	c = c_;
	h = h_;
	w = w_;
	 
	elements = n * c * h * w;
	bytes =  elements * byte_per_element;
	bad_flag = true;
	cudaError_t err = cudaMalloc(&gpu_data, bytes);
	if (err != cudaSuccess) {
		cout << "Error: Try to allocate " << bytes << " bytes of GPU memory failed in FloatTensor4D.constructor! Error code : " << err << endl;
		elements = 0;
		return false;
	}
	if (CUDNN_STATUS_SUCCESS != cudnnSetTensor4dDescriptor(desc, data_format, data_type, n, c, h, w)) {
		elements = 0;
		return false;
	}
	if (cudaSuccess != cudaMemset(gpu_data, 0, bytes)) { 
		return false;
	}
	bad_flag = false;
	return true;
}
const CudaTensor& CudaTensor::operator=(const CudaTensor& right) {
	byte_per_element = right.byte_per_element;
	data_format = right.data_format;
	data_type = right.data_type;
	if (!Init(right.n, right.c, right.h, right.w)) {
		bad_flag = true;
		return *this;
	}
	//TODO: different datatype
	bad_flag = (cudaSuccess != cudaMemcpy(gpu_data, right.gpu_data, bytes, cudaMemcpyDeviceToDevice));
	return *this;
}
const CudaTensor& CudaTensor::operator=(float val) {
	if (elements > 0) {
		if (val == 0.0) {
			bad_flag = cudaSuccess != cudaMemset(gpu_data, 0, bytes);
		}
		else {
			void *ptr = &val;
			__half val_h;
			if (data_type == CUDNN_DATA_HALF) {
				val_h = __float2half(val);
				ptr = &val_h;
			}
			bad_flag = CUDNN_STATUS_SUCCESS != cudnnSetTensor(GetCUDNNHandle(), desc, gpu_data, &ptr);
			
		}

	}
	return *this;
}
bool CudaTensor::Push(const float* cpu_data, int pos, int length) {
	if (pos < 0 ) return false;
	if (length < 0 || (length + pos > elements)) length = elements - pos;
 
	float* buffer = nullptr;
	bool r = true;
	cudaError_t e; 
	float one = 1.0f, zero = 0.0f;
	try{
		if (CUDNN_DATA_HALF == data_type) {  
			e = cudaMalloc(&buffer, length * sizeof(float));
			if (cudaSuccess != e) throw (int)e;
			e = cudaMemcpy(buffer, cpu_data, length * sizeof(float), cudaMemcpyHostToDevice);
			if (cudaSuccess != e) throw (int)e;
			r = f32_to_f16(reinterpret_cast<__half *>(gpu_data) + pos, buffer, length);
		}
		else {
			float* dst = reinterpret_cast<float *>(gpu_data) + pos;
			e = cudaMemcpy(dst, cpu_data, length * sizeof(float), cudaMemcpyHostToDevice);
			if (cudaSuccess != e) throw (int)e;
		}		
			
	}
	catch(int err){
		cerr << "Error: CudaTensor.Push ret " << err << endl;
		r = false;
	}
	if (buffer) cudaFree(buffer);
	return r;
}
bool CudaTensor::Push(const char* cpu_data, const tensor_data_header& header) {
	return true;
}
bool CudaTensor::Pull(float * cpu_data, int pos, int length) const {
	if (pos < 0) return false;
	if (length < 0 || (length + pos > elements)) length = elements - pos;
 
	float* buffer = nullptr;
	bool r = true;
	cudaError_t e; 
	float one = 1.0f, zero = 0.0f;
	try {
		if (CUDNN_DATA_HALF == data_type) {

			e = cudaMalloc(&buffer, length * sizeof(float));
			if (cudaSuccess != e) throw (int)e;			

			r = f16_to_f32(buffer, reinterpret_cast<__half *>(gpu_data) + pos, length);
			if (r) {
				e = cudaMemcpy( cpu_data, buffer, length * sizeof(float), cudaMemcpyDeviceToHost);
				if (cudaSuccess != e) throw (int)e;
			}
			
		}
		else {
			float* src = reinterpret_cast<float *>(gpu_data) + pos;
			e = cudaMemcpy(cpu_data, src, length * sizeof(float), cudaMemcpyHostToDevice);
			if (cudaSuccess != e) throw (int)e;
		}

	}
	catch (int err) {
		cerr << "Error: CudaTensor.Push ret " << err << endl;
		r = false;
	}
	if (buffer) cudaFree(buffer);
	return r;
}

bool CudaTensor::Concat(const vector<const CudaTensor*>& src) {
	int copied = 0;
	for (int i = 0; i < n; i++) {
		char* dest_mem = BatchData(i);
		for (int j = 0; j < (int)src.size(); j++) {
			const CudaTensor* s = src[j];
			if (h != s->h || w != s->w) return false;
			char *src_mem = s->BatchData(i);
			int copy_bytes = s->Elements3D() * s->byte_per_element;
			if (cudaSuccess != cudaMemcpy(dest_mem, src_mem, copy_bytes, cudaMemcpyDeviceToDevice)) return false;
			dest_mem += copy_bytes;
			copied += copy_bytes;
			if (copied > bytes) {
				cerr << " Error: CudaTensor.Concat overflow!\n";
				return false;
			}
		}
	}
	return true;
}

bool CudaTensor::Split(const vector<CudaTensor*>& dest) const {
	int copied = 0;
	for (int i = 0; i < n; i++) {
		char* src_mem = BatchData(i);
		for (int j = 0; j < (int)dest.size(); j++) {
			CudaTensor* s = dest[j];
			if (h != s->h || w != s->w) return false;
			char *dest_mem = s->BatchData(i);
			int copy_bytes = s->Elements3D() * s->byte_per_element;
			if (cudaSuccess != cudaMemcpy(dest_mem, src_mem, copy_bytes, cudaMemcpyDeviceToDevice)) return false;
			src_mem += copy_bytes;
			copied += copy_bytes;
			if (copied > bytes) {
				cerr << " Error: CudaTensor.Concat overflow!\n";
				return false;
			}
		}
	}
	return true;
}
bool CudaTensor::Release() {
	cudaError_t err = cudaSuccess;
	if (gpu_data) {
		err = cudaFree(gpu_data);
		gpu_data = nullptr;

	}
	n = 0;
	c = 0;
	h = 0;
	w = 0;
	bytes = 0;
	elements = 0;
	return cudaSuccess == err; 
}

bool CudaTensor::Randomize() {
	if (elements == 0) return true;
	unique_ptr<float> ptr( New float[elements]);
	float* buffer = ptr.get();
	for (int i = 0; i < elements; i++) {
		buffer[i] = rand_uniform_strong(-0.5, 0.5);
	}
	if (data_type == CUDNN_DATA_FLOAT) {
		return (cudaSuccess == cudaMemcpy(gpu_data, buffer, bytes, cudaMemcpyHostToDevice));
	}
	CudaPtr<float> gpu_buffer(elements, buffer);
	return f32_to_f16(reinterpret_cast<__half*>(gpu_data), gpu_buffer, elements);

}