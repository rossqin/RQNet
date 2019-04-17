#pragma once
#include "tinyxml2.h"
#include "cuda_tensor.h"
#include "activation.h"
using namespace tinyxml2;
class ParamPool;
class InferenceModule;
struct ForwardContext;
class Layer;
class CNNNetwork;
typedef map<string, InferenceModule*> ModulePool;
class InferenceModule {
protected:
	int index;
	int output_port;
	int input_height;
	int input_width;
	int output_height;
	int output_width;
	int input_channels;
	int output_channels; 
	 
	InferenceModule* logical_prev; 
	Layer* layer;
	CNNNetwork* network;
	
	vector<InferenceModule*> prevs;
	void GetPrevModules(const XMLElement* element);	 
	virtual bool Resize(int w, int h) = 0;
	bool PrepareShortcutDelta();
	void WritePorts(ofstream& xml) const;
public:
	string name;
	CudaTensor input, output;
	CudaTensor shortcut_delta;
	InferenceModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev);
	virtual ~InferenceModule() {};
	static InferenceModule* FromXmlElement(const XMLElement* element,  Layer* layer, CNNNetwork* network, InferenceModule* prev);
	inline int GetOutputChannels() const { return output_channels; }
	bool UpdateShortcutDelta(const CudaTensor& delta);
	virtual bool Forward(ForwardContext& context) ;
	virtual bool Backward(CudaTensor& delta) ;
	virtual bool UpdateParams(float lr) { return true; }
	virtual bool DistributeDeltas(CudaTensor& delta);
	virtual bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, bool fp16) const;
	virtual uint32_t GetFlops() const = 0;
};
class BatchNormModule;
class ConvolutionalModule : public InferenceModule {
protected:
	CudaTensor w;
	CudaTensor dw;
	//TODO: think about if not followed by batchnorm module.
	CudaTensor bias;
	CudaTensor dbias; 
	friend class BatchNormModule;

	int padding_w;
	int padding_h;
	int stride_w;
	int stride_h;
	int dilation_w;
	int dilation_h;
	
	cudnnFilterDescriptor_t w_desc;
	cudnnConvolutionDescriptor_t conv_desc;

	cudnnConvolutionFwdAlgo_t fwd_algo;
	cudnnConvolutionBwdDataAlgo_t bwdd_algo;
	cudnnConvolutionBwdFilterAlgo_t bwdf_algo; 
	bool Resize(int w, int h);
	friend class BatchNormModule;
public :
	~ConvolutionalModule();
	ConvolutionalModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev);
	bool Forward(ForwardContext& context);
	bool Backward(CudaTensor& delta);
	bool UpdateParams(float lr);
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, bool fp16) const;
	uint32_t GetFlops() const;
};
 
class PoolingModule : public InferenceModule {
protected:
	int stride_w;
	int stride_h;
	int window_w;
	int window_h;
	int pad_w;
	int pad_h;
	//int* indexes;
	CudaTensor stride_one_output;
	cudnnPoolingDescriptor_t desc;
	cudnnPoolingMode_t mode;
	bool Resize(int w, int h);
public :
	PoolingModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev);
	~PoolingModule();
	bool Forward(ForwardContext& context);
	bool Backward(CudaTensor& delta);
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, bool fp16) const;
	uint32_t GetFlops() const;
};
class BatchNormModule : public InferenceModule {
protected:
	bool fused;
	CudaTensor params; 
	CudaTensor training_params;
	bool freezed;
	cudnnTensorDescriptor_t t_desc; 
	bool Resize(int w, int h);
public:

	BatchNormModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev);
	~BatchNormModule();	
	bool Forward(ForwardContext& context);
	bool Backward(CudaTensor& delta);
	bool UpdateParams(float lr);
	//Do nothing.
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, bool fp16) const { return true; }
	uint32_t GetFlops() const;
	bool Fuse();
};
class ActivationModule : public InferenceModule {
protected:
	ActivationMode mode;
	float factor; 
	bool Resize(int w, int h);
public:
	ActivationModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev);
	~ActivationModule();
	bool Forward(ForwardContext& context);
	bool Backward(CudaTensor& delta);
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, bool fp16) const;
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
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, bool fp16) const;
	uint32_t GetFlops() const;
};


class DeconvModule : public InferenceModule {
protected:
 
public:
	DeconvModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev);
	~DeconvModule();
	bool Forward(ForwardContext& context);
	bool Backward(CudaTensor& delta);
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, bool fp16) const { return false; }
	uint32_t GetFlops() const { return 0; }
};