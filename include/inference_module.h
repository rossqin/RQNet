#pragma once
#include "tinyxml2.h"
#include "cuda_tensor.h"
#include "activation.h"
#include "OpenVINO.h"
using namespace tinyxml2;
class ParamPool;
class InferenceModule;

struct ForwardContext;
class Layer;
class CNNNetwork;
typedef map<string, InferenceModule*> ModulePool; 
struct PrevModuleOuput {
	InferenceModule* module;
	int group_id;
};
class EltwiseModule;
class ConcatModule;
class SplitModule;
class InferenceModule {
protected:
	int input_height;
	int input_width;
	int output_height;
	int output_width;
	int input_channels;
	int output_channels;
	bool concat_prevs;
	string name;
	CudaTensor input, output, shortcut_delta;
	InferenceModule* logical_prev;
	Layer* layer;
	CNNNetwork* network;
	vector<PrevModuleOuput> prevs; 

	// for OpenVINO IR
	const char* ir_type;
	int ir_output_port; 
	map<string, string> ir_params;


	friend class EltwiseModule;
	friend class ConcatModule;
	friend class SplitModule;

	static InferenceModule* last_parsing;
public: 
	vector<bool> valid_channels; // for network slimming
	virtual ~InferenceModule() {}
	virtual bool Forward(ForwardContext& context) ;
	virtual bool Backward(CudaTensor& delta) ;
	virtual bool UpdateParams(float lr) { return true; }
	virtual bool DistributeDeltas(CudaTensor& delta);
	virtual bool RenderOpenVINOIR(vector<OpenVINOIRv7Layer>& layers, vector<OpenVINOIRv7Edge>& edges,
		ofstream& bin, size_t& bin_offset, bool fp16);
	virtual void WriteOpenVINOOutput(ofstream& xml) const {} 
	virtual uint32_t GetFlops() const = 0;
	virtual void ParsePrevModules(const XMLElement* element);
	virtual bool Resize(int w, int h) = 0; 
	virtual CudaTensor& GetOutput(int gid) { return output; }
	virtual CudaTensor& GetShortcutDelta(int gid) { return shortcut_delta; }
	virtual bool ShortcutDelta(const CudaTensor& d, int group_id = -1);
	virtual int OutputCount() const { return 1; }
	virtual bool CheckRedundantChannels(float c_threshold, float w_threshold);
	virtual bool Prune() { return true; }
	virtual void SyncValidChannels(vector<bool>& vc, int s, int n, int g = -1);

	InferenceModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev);
	inline const char* Precision() const { return (output.DataType() == CUDNN_DATA_FLOAT) ? "FP32" : "FP16"; }
	inline int GetOutputChannels() const { return output_channels; } 
	inline const string& Name() const { return name; }
	inline int PrevCount() const { return prevs.size();  }
	static InferenceModule* FromXmlElement(const XMLElement* element, Layer* layer, CNNNetwork* network, InferenceModule* prev);

	//bool PrepareShortcutDelta();
	InferenceModule* GetPrev(int n, int& group_id, bool ignore_bn = true) const;

};
class BatchNormModule;
class ConvolutionalModule : public InferenceModule {
protected:
	int groups;
	CudaTensor* forward_input;
	CudaTensor w;
	CudaTensor dw;

	//for adam
	CudaTensor adam_m;
	CudaTensor adam_v; 
	CudaTensor adam_bias_m;
	CudaTensor adam_bias_v;

	//TODO: think about if not followed by batchnorm module.
	CudaTensor bias;
	CudaTensor dbias; 
	BatchNormModule* following_bn;
	friend class BatchNormModule;

	int padding_w;
	int padding_h;
	int stride_w;
	int stride_h;
	int dilation_w;
	int dilation_h; 

	size_t workspace_size;
	
	cudnnFilterDescriptor_t w_desc;
	cudnnConvolutionDescriptor_t conv_desc;

	cudnnConvolutionFwdAlgo_t fwd_algo;
	cudnnConvolutionBwdDataAlgo_t bwdd_algo;
	cudnnConvolutionBwdFilterAlgo_t bwdf_algo; 
	vector<bool> valid_in_channels;
	bool Resize(int w_, int h);
	friend class BatchNormModule;
	friend class CNNNetwork;
	friend class InferenceModule;
public :
	~ConvolutionalModule();
	ConvolutionalModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev);
	bool Forward(ForwardContext& context);
	bool Backward(CudaTensor& delta);
	bool UpdateParams(float lr);
	bool RenderOpenVINOIR(vector<OpenVINOIRv7Layer>& layers, vector<OpenVINOIRv7Edge>& edges, 
		ofstream& bin, size_t& bin_offset, bool fp16) ;
	uint32_t GetFlops() const;
	bool CheckRedundantChannels(float c_threshold, float w_threshold);
	bool Prune();
};
 
class PoolingModule : public InferenceModule {
protected:
	int stride_w;
	int stride_h;
	int window_w;
	int window_h;
	int pad_wl;
	int pad_wr;
	int pad_ht;
	int pad_hb;
	int* indexes;
	cudnnPoolingMode_t mode; 
	bool Resize(int w, int h);
public :
	PoolingModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev);
	~PoolingModule();
	bool Forward(ForwardContext& context);
	bool Backward(CudaTensor& delta);
	uint32_t GetFlops() const;
};
class BatchNormModule : public InferenceModule {
protected:
	bool fused;
	CudaTensor params; 
	CudaTensor training_params;

	CudaTensor adam_params; 

	bool freezed;
	cudnnTensorDescriptor_t t_desc; 
	CudaTensor* forward_input;
	bool Resize(int w, int h);
	friend class ConvolutionalModule;
public:

	BatchNormModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev);
	~BatchNormModule();	
	bool Forward(ForwardContext& context);
	bool Backward(CudaTensor& delta);
	bool UpdateParams(float lr);
	inline bool IsFused() const { return fused; } 
	uint32_t GetFlops() const;
	bool Fuse();
	bool CheckRedundantChannels( float c_threshold, float w_threshold);
	bool Prune();
};
class ActivationModule : public InferenceModule {
protected:
	ActivationMode mode;
	float factor; 
	bool Resize(int w, int h);
public:
	ActivationModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev); 
	bool Forward(ForwardContext& context);
	bool Backward(CudaTensor& delta);
	uint32_t GetFlops() const;
};
class UpSampleModule : public InferenceModule {
protected:
	int stride_w;
	int stride_h; 
	bool Resize(int w, int h);
public:
	UpSampleModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev);
	~UpSampleModule();
	bool Forward(ForwardContext& context);
	bool Backward(CudaTensor& delta);  
	uint32_t GetFlops() const;
};
 
class EltwiseModule : public InferenceModule {
public:
	enum { SUM = 0, SUB, MUL, DIV, MAX, MIN, SQUARED_DIFF, FLOOR_MOD, POW, LOGICAL_AND, LOGICAL_OR, LOGICAL_XOR, LESS, LESS_EQUAL, GREATER, GREATER_EQUAL, EQUAL, NOT_EQUAL };
protected:
	int operation;
	bool Resize(int w, int h);
public:
	
	EltwiseModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev); 
	bool Forward(ForwardContext& context); 
	bool Backward(CudaTensor& delta);
	uint32_t GetFlops() const;
	bool CheckRedundantChannels(float c_threshold, float w_threshold);
};

class SSDModule : public InferenceModule {
protected:
	bool focal_loss;
	int classes;
	float ignore_thresh;
	float truth_thresh;
	int  s_k;
	vector<float> def_boxes;
	bool Resize(int w, int h);
public:
	SSDModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev);
	~SSDModule();
	bool Forward(ForwardContext& context);
	bool Backward(CudaTensor& delta); 
	uint32_t GetFlops() const { return 0; }
};
// just for placeholder 
class ConcatModule : public InferenceModule { 
public:
	ConcatModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev);
	uint32_t GetFlops() const { return 0; }
	bool Resize(int w, int h);
	bool Forward(ForwardContext& context);
	bool Backward(CudaTensor& delta); 
	 
};
class SplitModule: public InferenceModule {
protected:
	int groups;
	CudaTensor* forward_input;
	vector<CudaTensor> outputs;
	vector<CudaTensor> deltas;
public:
	SplitModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev);
	uint32_t GetFlops() const { return 0; }
	bool Resize(int w, int h);
	bool Forward(ForwardContext& context); 
	bool Backward(CudaTensor& delta);
	CudaTensor& GetOutput(int gid) { if (gid < 0 || gid >= outputs.size()) return *forward_input; return outputs[gid]; }  
	CudaTensor& GetShortcutDelta(int gid) { return deltas[gid]; }
	bool ShortcutDelta(const CudaTensor& d, int group_id);
	int OutputCount() const { return groups; }
	bool CheckRedundantChannels(float c_threshold, float w_threshold);
	void SyncValidChannels(vector<bool>& vc, int s, int n, int g );
};

class ShuffleModule :public InferenceModule {
protected:
	int groups;
public:
	ShuffleModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev);
	uint32_t GetFlops() const { return 0; }
	bool Resize(int w, int h);
	bool Forward(ForwardContext& context);
	bool Backward(CudaTensor& delta);

};
class ClassifierModule : public InferenceModule {
protected:
public:
	ClassifierModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev);
	uint32_t GetFlops() const { return 0; }
	bool Resize(int w, int h) { return true; }
	bool Forward(ForwardContext& context);
	bool Backward(CudaTensor& delta);
};