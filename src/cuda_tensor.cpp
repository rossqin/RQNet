#include "stdafx.h"
#include "cuda_tensor.h"
#include "network.h"
#include <memory>

CudaTensor::CudaTensor(const CudaTensor & right) {
	data_format = right.data_format;
	data_type = right.data_type;
	byte_per_element = right.byte_per_element;
	dims = right.dims;
	gpu_data = nullptr;
	elements = right.elements;
	bytes = right.bytes;
	desc = nullptr;
	bad_flag = true;
	cudnnCreateTensorDescriptor(&desc);
	if(CUDNN_STATUS_SUCCESS != cudnnSetTensor4dDescriptor(desc, data_format, data_type, dims.n, dims.c, dims.h, dims.w))
		return ;
	cudaMalloc(&gpu_data, bytes);
	bad_flag = (cudaSuccess != cudaMemcpy(gpu_data, right.gpu_data, bytes, cudaMemcpyDeviceToDevice)); 
}

CudaTensor::CudaTensor(cudnnDataType_t t, cudnnTensorFormat_t f) {
	dims = { 0 };
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
	if(!gpu_data || b < 0 || b>= dims.n) return nullptr;
	return reinterpret_cast<char*>(gpu_data) + b *(Elements3D() * byte_per_element);
}
bool CudaTensor::Init(const TensorDims& d) {
	
	if (dims == d) {
		if (bytes != elements * byte_per_element) {
			bytes = elements * byte_per_element;
			cudaFree(gpu_data);
			cudaMalloc(&gpu_data, bytes);
		}
		return cudaSuccess == cudaMemset(gpu_data, 0, bytes);
	}
	if (gpu_data) { 
		cudaError_t err = cudaFree(gpu_data);
		gpu_data = nullptr;
		if (err != cudaSuccess) {
			cerr << " Error: cudaFree at 0x" << hex << setw(10) << setfill('0') << (size_t)gpu_data << " return " << (int)err << endl;
		}
	}
	if (nullptr == desc) {
		if(CUDNN_STATUS_SUCCESS != cudnnCreateTensorDescriptor(&desc))
			return false;
	}
	dims = d;
	 
	elements = d.n * d.c * d.h * d.w;
	bytes =  elements * byte_per_element;
	bad_flag = true;
	cudaError_t err = cudaMalloc(&gpu_data, bytes);
	if (err != cudaSuccess) { 
		cerr << " Error: Try to allocate " << bytes << " bytes of GPU memory failed in CudaTensor.Init! Error code : " << err << endl;
		elements = 0;
		return false;
	}
	if (CUDNN_STATUS_SUCCESS != cudnnSetTensor4dDescriptor(desc, data_format, data_type, d.n, d.c, d.h, d.w)) {
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
	if (!Init(right.dims)) {
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
	if (0 == elements) return false;
	int stored_b = header.dims[0], stored_c, stored_h, stored_w;
	cudnnDataType_t stored_t = (header.data_type == CUDNN_DATA_DOUBLE) ? CUDNN_DATA_FLOAT : header.data_type;
	if (data_format == CUDNN_TENSOR_NHWC) { 
		stored_c = header.dims[1];
		stored_h = header.dims[2];
		stored_w = header.dims[3];
	}
	else {
		stored_h = header.dims[2];
		stored_w = header.dims[3];
		stored_c = header.dims[1];
	} 
	int stored_elements = stored_b * stored_c * stored_h * stored_w;
	CudaPtr<char> temp(header.bytes , cpu_data);	
	if (stored_elements >= elements) {
		if (data_type == stored_t) {
			return cudaSuccess == cudaMemcpy(gpu_data, temp, bytes, cudaMemcpyDeviceToDevice);
		}
		else if(data_type == CUDNN_DATA_HALF) {
			return f32_to_f16(reinterpret_cast<__half*>(gpu_data), reinterpret_cast<float*>(temp.ptr), elements);
		}
		else {
			return f16_to_f32(reinterpret_cast<float*>(gpu_data), reinterpret_cast<__half*>(temp.ptr), elements);
		}
	}
	else {
		if (data_type == stored_t) {
			if(cudaSuccess != cudaMemcpy(gpu_data, temp, header.bytes, cudaMemcpyDeviceToDevice)) return false;
			if (stored_w == dims.w && stored_h == dims.h) {
				int left = bytes - header.bytes;
				char* dst = reinterpret_cast<char*>(gpu_data) + header.bytes;
				while (left > 0) {
					int to_copy = min((int)header.bytes, left);
					if (cudaSuccess != cudaMemcpy(dst, temp, to_copy, cudaMemcpyDeviceToDevice)) return false;
					dst += to_copy;
					left -= to_copy;
				}
				return true;
			}
		}
		else if (data_type == CUDNN_DATA_HALF) {
			if(!f32_to_f16(reinterpret_cast<__half*>(gpu_data), reinterpret_cast<float*>(temp.ptr), stored_elements))
				return false;
		}
		else {
			if(!f16_to_f32(reinterpret_cast<float*>(gpu_data), reinterpret_cast<__half*>(temp.ptr), stored_elements))
				return false;
		}
		int left = elements - stored_elements;
		CpuPtr<float> ptr(left); 
		float* buffer = ptr.ptr;
		for (int i = 0; i < left; i++) {
			buffer[i] = rand_uniform_strong(-0.1f, 0.1f);
		}
		if (data_type == CUDNN_DATA_FLOAT) {
			float* dst = reinterpret_cast<float*>(gpu_data) + stored_elements;
			return (cudaSuccess == cudaMemcpy(dst, buffer, left * sizeof(float), cudaMemcpyHostToDevice));
		}
		CudaPtr<float> gpu_buffer(left, buffer);
		return f32_to_f16(reinterpret_cast<__half*>(gpu_data), gpu_buffer, left);
	}
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
			e = cudaMemcpy(cpu_data, src, length * sizeof(float), cudaMemcpyDeviceToHost);
			if (cudaSuccess != e) throw (int)e;
		}

	}
	catch (int err) {
		cerr << "Error: CudaTensor.Pull ret " << err << endl;
		r = false;
	}
	if (buffer) cudaFree(buffer);
	return r;
}

bool CudaTensor::Concat(const vector<const CudaTensor*>& src) {
	int copied = 0;
	for (int i = 0; i < dims.n; i++) {
		char* dest_mem = BatchData(i);
		for (int j = 0; j < (int)src.size(); j++) {
			const CudaTensor* s = src[j];
			if (dims.h != s->dims.h || dims.w != s->dims.w) {
				return false;
			}
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
	for (int i = 0; i < dims.n; i++) {
		char* src_mem = BatchData(i);
		for (int j = 0; j < (int)dest.size(); j++) {
			CudaTensor* s = dest[j];
			if (dims.h != s->dims.h || dims.w != s->dims.w) return false;
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
	dims = { 0 };
	bytes = 0;
	elements = 0;
	return cudaSuccess == err; 
}

bool CudaTensor::Randomize() {
	if (elements == 0) return true;
	unique_ptr<float> ptr( New float[elements]);
	uniform_real_distribution<double> uniform(0, 0.25f / elements);
	default_random_engine engine;
	engine.seed(chrono::system_clock::now().time_since_epoch().count());

	float* buffer = ptr.get();
	for (int i = 0; i < elements; i++) {
		buffer[i] = uniform(engine);
	}
	if (data_type == CUDNN_DATA_FLOAT) {
		return (cudaSuccess == cudaMemcpy(gpu_data, buffer, bytes, cudaMemcpyHostToDevice));
	}
	CudaPtr<float> gpu_buffer(elements, buffer);
	return f32_to_f16(reinterpret_cast<__half*>(gpu_data), gpu_buffer, elements);

}

bool CudaTensor::Save(const char * filename, int batch) {
	if (bytes == 0) return false;
	ofstream f(filename, ios::binary | ios::trunc);
	if (!f.is_open()) return false;
	int write_len = bytes;
	char* gpu_buffer = reinterpret_cast<char*>(gpu_data);
	if (batch >= 0 && batch < dims.n) {
		write_len = dims.c * dims.h * dims.w * byte_per_element;
		gpu_buffer += (batch * write_len);
	}
	char* buffer = New char[write_len];
	unique_ptr<char> temp(buffer);
	cudaError_t e = cudaMemcpy(buffer, gpu_buffer, write_len, cudaMemcpyDeviceToHost);
	if (e != cudaSuccess) {
		cerr << "Failed to save file `" << filename << "` : Error copy data from GPU.\n";
		f.close();
		return false;
	}
	f.write(buffer, write_len);
	f.close();
	return true;
}

bool CudaTensor::DisplayInFile(const char * filename, int batch) {
	ofstream f(filename, ios::trunc);
	if (!f.is_open()) return false;
	int batch_start = 0;
	int batch_stop = dims.n;
	if (batch >= 0 && batch < dims.n) {
		batch_start = batch;
		batch_stop = batch + 1;
	}
	int c_elem = dims.h * dims.w;
	int b_elem = dims.c * c_elem;
	CpuPtr<float> buffer_cpu(b_elem);
	 

	if (data_type == CUDNN_DATA_FLOAT)
		f << "FP32 [" << dims.n << ", " << dims.c << ", " << dims.h << ", " << dims.w << "]\n";
	else
		f << "FP16 [" << dims.n << ", " << dims.c << ", " << dims.h << ", " << dims.w << "]\n";
	
	
	for (int b = batch_start; b < batch_stop; b++) {
		f << "batch: " << b << endl;
		if (data_type == CUDNN_DATA_FLOAT) {
			float* src = reinterpret_cast<float*>(gpu_data) + b * b_elem;
			if (cudaSuccess != cudaMemcpy(buffer_cpu.ptr, src, b_elem * sizeof(float), cudaMemcpyDeviceToHost)) {
				return false;
			}
		}
		else {
			CudaPtr<float> buffer(b_elem);
			__half* src = reinterpret_cast<__half*>(gpu_data) + b * b_elem;
			if (!f16_to_f32(buffer, src, b_elem))
				return false;
			if (cudaSuccess != cudaMemcpy(buffer_cpu.ptr, buffer.ptr, b_elem * sizeof(float), cudaMemcpyDeviceToHost)) {
				return false;
			}

		} 
		for (int i = 0; i < b_elem; i++) {
			f << fixed << setprecision(4) << buffer_cpu[i] << " "; 
			if (dims.w > 1) {
				if ((i + 1) % dims.w == 0)
					f << endl;
				if (dims.h > 1) {
					if ((i + 1) % c_elem == 0)
						f << endl;
				}
			} 
		}
		f << endl;

	}
	f.close(); //*/
	return true;
}

bool CudaTensor::Cache(char *& cpu_data) {
	if (!gpu_data || 0 == bytes) return false;
	if (cpu_data) {
		cpu_data = (char*)realloc(cpu_data, bytes);
	}
	else
		cpu_data = New char[bytes]; 
	if (cudaSuccess != cudaMemcpy(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost)) {
		return false;
	}
	bool b = cudaSuccess == cudaFree(gpu_data);
	if(b)
		gpu_data = nullptr;
	return b;
}

bool CudaTensor::Restore(char *& cpu_data) {
	if (gpu_data || 0 == bytes) return false;
	if (cudaSuccess != cudaMalloc(&gpu_data, bytes))
		return false;
	cudaError_t e = cudaMemcpy(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice);
	if (cudaSuccess != e) {
		return false;
	} 
	delete[]cpu_data;
	cpu_data = nullptr;
	return true;
}
