#pragma once
 
enum TensorOrder {
	TO_NCHW = 0, /* row major (wStride = 1, hStride = w) */
	TO_NHWC = 1, /* feature maps interleaved ( cStride = 1 )*/
};

enum DataType {
	DT_INVALID = 0,
	DT_FLOAT32,
	DT_FLOAT16,
	DT_INT32,
	DT_UINT32,
	DT_INT16,
	DT_UINT16,
	DT_DOUBLE,
	DT_UINT8,
	DT_STRING
};
enum TensorOrder;
class FloatTensor4D {
protected: 
	TensorOrder order;
	
	float* gpu_data; 
	bool fine;
	int batch;
	int channels;
	int height;
	int width;
	int elements_2d;
	int elements_3d;
	int elements;
	int bytes;
	
public:
	FloatTensor4D();
	FloatTensor4D(const FloatTensor4D& right);
	virtual ~FloatTensor4D();
	bool Init(int b, int c, int w, int h, TensorOrder o);
	inline bool InitFrom(const FloatTensor4D& right){ return Init(right.batch, right.channels, right.width, right.height, right.order); }
	bool Release();
	inline operator float*() const{ return gpu_data; }
	inline float* GetMem() const { return gpu_data; }
	inline int MemBytes() const { return bytes; }
	inline int MemElements() const { return elements; }

	inline int Elements2D() const { return elements_2d; }
	inline int Elements3D() const { return elements_3d; }

	inline TensorOrder GetOrder() const { return order; }

	inline int GetBatch() const { return batch; }
	inline int GetChannels() const { return channels; }
	inline int GetHeight() const { return height; }
	inline int GetWidth() const { return width; } 

	inline bool IsEmpty() const { return NULL == gpu_data; }

	const FloatTensor4D& operator=(const FloatTensor4D& right);
	const FloatTensor4D& operator=(float val);

	bool SameDememsion(const FloatTensor4D& right) const;

	bool Add(const FloatTensor4D& right);
	bool Add(const float*vals, size_t length);
	bool Add(float val);
	bool Mul(float val);
	bool MulAdd(float scale, float bias);
	bool MulAdd(float scale, const FloatTensor4D& right);
	bool AddScale(const FloatTensor4D& right, float scale);

	bool Set3DData(int index, const float*  src, bool src_from_cpu = false);
	bool Get3DData(int index, float*  out, bool to_cpu = true) const;

	bool Concat(int b, int c, const float* src, int src_c, int src_w, int src_h);

	void DumpToFile(const string& filename, int b = 0 , int c = 0) const ;

	bool UpSample(FloatTensor4D& result, int stride_w, int stride_h ) const;
	bool DownSample(FloatTensor4D& result, int stride_w, int stride_h ) const;

	bool Randomize(float min_ = 0.0f, float max_ = 1.0f);

	bool SaveBatchData(const string& filename, int b = 0);
 
#if 0
	float* MoveDataToCPU();
	float* RestoreDataFromCPU();
#endif
	bool CopyDataFromCPU(void * data, int data_bytes, DataType data_type, uint16_t dims[4]); 
	char* CopyToCPU() const ;

};
bool add_in_gpu(float* dest, const float* src, int elements);
template <typename T>
class CudaPtr {
protected:
	T*  ptr;
	int bytes;
	mutable cudaError_t err;
	CudaPtr() = default;
public:	
	cudaError_t GetError() const { return err; }
	operator T*() { return ptr; }
	bool ToCPU(void* dest, int length = -1) const {
		if (length <= 0) length = bytes;
		err = cudaMemcpy(dest, ptr, length, cudaMemcpyDeviceToHost);
		return err == cudaSuccess;
	}
	CudaPtr(int length, void* src = NULL) {
		ptr = NULL;
		bytes = length * sizeof(T);
		err = cudaMalloc(&ptr, bytes);
		if (ptr && src) {
			err = cudaMemcpy(ptr, src, bytes, cudaMemcpyHostToDevice);
		}
	}
	~CudaPtr() { if (ptr) cudaFree(ptr); } 
};