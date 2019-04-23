#pragma once

#pragma pack(push, 1)

struct param_file_header {
	uint16_t version_major; // 1
	uint16_t version_minor; // 0 
	cudnnTensorFormat_t tensor_order; // NCHW by default
	uint32_t reserved; // 0 
	uint32_t seen;
	uint32_t strings_bytes;
};
// file layout :
// weights_file_header 
// n bytes of string data
// weights data 
#pragma pack(pop)
class CudaTensor;
class ParamPool {
protected:
	bool release_mem;
	map<string, CudaTensor*> uninit_params;
	map<string, CudaTensor*> params;
	cudnnTensorFormat_t tensor_order; // TO_NCHW by default
	uint32_t iteration;
public:
	
	ParamPool() { tensor_order = CUDNN_TENSOR_NCHW; release_mem = false; }
	~ParamPool();
	inline uint32_t GetIteration() const { return iteration; }
	void Put(string key, CudaTensor* tensor);
	CudaTensor * GetParameter(const string& key);
	bool Load(const char* filename);
	bool Save(const char* filename,int i = -1);
	bool TransformDarknetWeights(const char* cfg, const char* filename,const char* out_dir);
	 

};
ParamPool& GetParamPool();