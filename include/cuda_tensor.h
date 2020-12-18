#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cudnn.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#pragma pack(push, 1)

// file layout :
// weights_file_header 
// n bytes of string data
// weights data
struct tensor_data_header {
	cudnnDataType_t data_type;
	uint16_t dims[4];
	uint16_t name_index;
	uint32_t bytes;
};
#pragma pack(pop)
template <typename T>
class CpuPtr {
protected:
	int len;
	int bytes;
	mutable cudaError_t err;
public:
	T*  ptr;
	cudaError_t GetError() const { return err; }
	operator T*() const { return ptr; }
	T& operator [](int i) { return ptr[i]; }
	inline void Reset() { memset(ptr, 0, bytes); }
	inline int Length() const { return len; }
	inline int Bytes() const { return bytes; }
	inline bool Pull(const void* gpu_data) { return cudaSuccess == cudaMemcpy(ptr, gpu_data, bytes, cudaMemcpyDeviceToHost); }
	CpuPtr(int length, const void* src = nullptr) {
		len = length;
		ptr = New T[length];
		err = cudaSuccess;
		bytes = length * sizeof(T); 
		if (src)
			err = cudaMemcpy(ptr, src, bytes, cudaMemcpyDeviceToHost);
		else
			memset(ptr, 0, bytes);
		 
	}
	~CpuPtr() { if (ptr) delete []ptr; }
};
template <typename T>
class CudaPtr {
protected:	
	int bytes;
	mutable cudaError_t err;
	CudaPtr() = default;
public:
	T*  ptr;
	cudaError_t GetError() const { return err; }
	inline int Bytes() const { return bytes; }
	operator T*() const { return ptr; }
	bool ToCPU(void* dest, int length = -1) const {
		if (length <= 0) length = bytes;
		err = cudaMemcpy(dest, ptr, length, cudaMemcpyDeviceToHost);
		return err == cudaSuccess;
	}
	CudaPtr(int length, const void* src = nullptr) {
		ptr = nullptr;
		bytes = length * sizeof(T);
		err = cudaMalloc(&ptr, bytes);
		if (ptr && src) {
			err = cudaMemcpy(ptr, src, bytes, cudaMemcpyHostToDevice);
		}
	}
	~CudaPtr() { if (ptr) cudaFree(ptr); }
};
struct TensorDims {
	int n, c, h, w;
};
inline bool operator==(const TensorDims& d1, const TensorDims& d2) { return d1.n == d2.n && d1.c == d2.c && d1.h == d2.h && d1.w == d2.w;}
inline bool operator!=(const TensorDims& d1, const TensorDims& d2) { return d1.h != d2.h || d1.w != d2.w || d1.c != d2.c || d1.n != d2.n ; }

class CudaTensor {
protected:
	void* gpu_data;
	TensorDims dims;
	cudnnTensorDescriptor_t desc;
	cudnnTensorFormat_t data_format;
	cudnnDataType_t data_type;
	int elements;
	int bytes;
	int byte_per_element;
	bool bad_flag;
	CudaTensor() {} 
public:
	CudaTensor(const CudaTensor& right);
	CudaTensor(cudnnDataType_t t , cudnnTensorFormat_t f );
	~CudaTensor();
	inline int Batch() const { return dims.n; }
	inline int Channel() const { return dims.c; }
	inline int Height() const { return dims.h; }
	inline int Width() const { return dims.w; }
	inline int Elements() const { return elements; }
	inline void* Data() const { return gpu_data; }
	inline TensorDims Dims() const { return dims; }
	inline void DataType(cudnnDataType_t t) { data_type = t; byte_per_element = (t == CUDNN_DATA_FLOAT) ? sizeof(float) : sizeof(__half); }
	inline int Bytes() const { return bytes; }
	inline int Elements2D() const { return (data_format == CUDNN_TENSOR_NCHW) ? dims.h * dims.w : dims.c * dims.w; }
	inline int Elements3D() const { return dims.h * dims.w * dims.c; }
	inline operator void*() const { return gpu_data; }
	inline int ElementBytes() const { return byte_per_element; }
	inline cudnnDataType_t DataType() const { return data_type; }
	inline cudnnTensorFormat_t DataFormat() const { return data_format; }
	inline bool Like(const CudaTensor& r) const { return dims == r.dims; }
	inline bool DifferentShape(const CudaTensor& r) const { return dims != r.dims; }
	char* BatchData(int b) const;
	inline cudnnTensorDescriptor_t Descriptor() const { return desc; }

	bool Init(const TensorDims& d);
	const CudaTensor& operator=(const CudaTensor& right);
	const CudaTensor& operator=(float val);

	bool Push(const float* cpu_data, int pos = 0, int length = -1);
	bool Push(const char* cpu_data, const tensor_data_header& header);
	bool Pull(float *cpu_data, int pos = 0, int length = -1) const ;
	bool Concat(const vector<const CudaTensor*>& src);
	bool Split(const vector<CudaTensor*>& dest) const;
	bool UpSample(CudaTensor& output, int stride_w, int stride_h);
	bool DownSample(CudaTensor& output, int stride_w, int stride_h);
	bool Add(const CudaTensor& op);
	bool Add(float op);
	bool MulAdd(float op_m, float op_a);
	bool MulAdd(const CudaTensor& op_m, const CudaTensor& op_a);
	bool Release();
	bool Randomize();
	bool Save(const char* filename, int batch = -1);
	bool DisplayInFile(const char* filename, int batch = -1);
	bool Cache(char*& cpu_data);
	bool Restore(char*& cpu_data);
};