#pragma once

#pragma pack(push, 1)

struct param_file_header {
	uint16_t version_major; // 1
	uint16_t version_minor; // 0 
	TensorOrder tensor_order; // TO_NCHW by default
	uint32_t reserved; // 0 
	uint32_t seen;
	uint32_t strings_bytes;
};
// file layout :
// weights_file_header 
// n bytes of string data
// weights data
struct tensor_data_header {
	DataType data_type;
	uint16_t dims[4];
	uint16_t name_index;
	uint32_t bytes; 
};
#pragma pack(pop)

class ParamPool {
protected:
	bool release_mem;
	map<string, FloatTensor4D*> uninit_params;
	map<string, FloatTensor4D*> params;
	TensorOrder tensor_order; // TO_NCHW by default
public:
	uint32_t iteration;
	ParamPool() { tensor_order = TO_NCHW; release_mem = false; }
	~ParamPool();
	inline uint32_t GetIteration() const { return iteration; }
	void Put(string key, FloatTensor4D* tensor);
	FloatTensor4D * GetParameter(const string& key);
	bool Load(const char* filename);
	bool Save(const char* filename);
	bool TransformDarknetWeights(const char* cfg, const char* filename,const char* out_filename);
	 

};
ParamPool& GetParamPool();