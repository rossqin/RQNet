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
	bool data_in_gpu;
	TensorOrder order;
	size_t bytes;
	size_t elements;
	float* gpu_data;
	float* cpu_data; // allow temporarily move data to cpu.
	bool fine;
	int batch;
	int channels;
	int height;
	int width;
	size_t elements_2d;
	size_t elements_3d;
public:
	FloatTensor4D(bool in_gpu = true);
	FloatTensor4D(const FloatTensor4D& right);
	virtual ~FloatTensor4D();
	virtual bool Init(int b, int c, int w, int h, TensorOrder o);
	bool Release();
	inline float* GetMem() const { return gpu_data; }
	inline size_t MemBytes() const { return bytes; }
	inline size_t MemElements() const { return elements; }

	inline size_t Elements2D() const { return elements_2d; }
	inline size_t Elements3D() const { return elements_3d; }

	inline TensorOrder GetOrder() const { return order; }

	inline int GetBatch() const { return batch; }
	inline int GetChannels() const { return channels; }
	inline int GetHeight() const { return height; }
	inline int GetWidth() const { return width; }

	inline bool DataInGPU() const { return data_in_gpu; }

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

	void DumpToFile(const string& filename, int b = 0 , int c = 0) const ;

	bool UpSample(FloatTensor4D& result, int stride_w, int stride_h ) const;
	bool DownSample(FloatTensor4D& result, int stride_w, int stride_h ) const;

	bool Randomize(float min_ = 0.0f, float max_ = 1.0f);
 

	float* MoveDataToCPU();
	float* RestoreDataFromCPU();
	bool CopyDataFromCPU(void * data, size_t data_bytes, DataType data_type, uint16_t dims[4]); 

};
bool add_in_gpu(float* dest, float* src, int elements);
