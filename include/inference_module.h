#pragma once
#include "tinyxml2.h"
#include "activation.h"
using namespace tinyxml2;
class ParamPool;
class InferenceModule;
struct ForwardContext;
class Layer;
class FloatTensor4D;
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
	cudnnTensorDescriptor_t x_desc, y_desc;
	Layer* layer;
	
	vector<InferenceModule*> prevs;
	void GetPrevModules(const XMLElement* element);	 
	virtual bool InitDescriptors() = 0;
	bool PrepareShortcutDelta();
	void WritePorts(ofstream& xml) const;
public:
	string name;
	FloatTensor4D input;
	FloatTensor4D output;
	FloatTensor4D shortcut_delta;
	InferenceModule(const XMLElement* element, Layer* l, InferenceModule* prev);
	virtual ~InferenceModule() {};
	static InferenceModule* FromXmlElement(const XMLElement* element,  Layer* layer, InferenceModule* prev);
	inline int GetOutputChannels() const { return output_channels; }
	bool UpdateShortcutDelta(const FloatTensor4D& delta);
	virtual bool Forward(ForwardContext& context) ;
	virtual bool Backward(FloatTensor4D& delta) ;
	virtual bool UpdateParams(float lr) { return true; }
	virtual bool DistributeDeltas(FloatTensor4D& delta);
	virtual bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, bool fp16) const;
	virtual uint32_t GetFlops() const = 0;
};
class BatchNormModule;
class ConvolutionalModule : public InferenceModule {
protected:
	FloatTensor4D w;
	FloatTensor4D dw;
	//TODO: think about if not followed by batchnorm module.
	FloatTensor4D bias;
	FloatTensor4D dbias;
	cudnnTensorDescriptor_t db_desc; 
	// for IR generation
	BatchNormModule* followed_bn_module;
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
	bool InitDescriptors();
	friend class BatchNormModule;
public :
	~ConvolutionalModule();
	ConvolutionalModule(const XMLElement* element, Layer* l, InferenceModule* prev);
	bool Forward(ForwardContext& context);
	bool Backward(FloatTensor4D& delta);
	bool UpdateParams(float lr);
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, bool fp16) const;
	uint32_t GetFlops() const;
};
enum PooingMode{
	POOLING_MAX = 0,
	POOLING_AVG 
};
class PoolingModule : public InferenceModule {
protected:
	int stride_w;
	int stride_h;
	int window_w;
	int window_h;
	int pad_w;
	int pad_h;
	int* indexes;
	PooingMode mode;
	bool InitDescriptors();
public :
	PoolingModule(const XMLElement* element, Layer* l, InferenceModule* prev);
	~PoolingModule();
	bool Forward(ForwardContext& context);
	bool Backward(FloatTensor4D& delta);
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, bool fp16) const;
	uint32_t GetFlops() const;
};
class BatchNormModule : public InferenceModule {
protected:
	FloatTensor4D params; 
	float* mu;
	float* var;
	float* gamma_update;
	float* beta_update;
	bool freezed;
	cudnnTensorDescriptor_t t_desc; 
	bool InitDescriptors();
public:
	BatchNormModule(const XMLElement* element, Layer* l, InferenceModule* prev);
	~BatchNormModule();	
	bool Forward(ForwardContext& context);
	bool Backward(FloatTensor4D& delta);
	bool UpdateParams(float lr);
	//Do nothing.
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, bool fp16) const { return true; }
	uint32_t GetFlops() const;
	bool CalcWeightsForIR(FloatTensor4D& weight, FloatTensor4D& bias, float epsilon = 1.0e-5);
};
class ActivationModule : public InferenceModule {
protected:
	ACTIVATION_TYPE atype;
	float factor; 
	bool InitDescriptors();
public:
	ActivationModule(const XMLElement* element, Layer* l, InferenceModule* prev);
	~ActivationModule(){ }
	bool Forward(ForwardContext& context);
	bool Backward(FloatTensor4D& delta);
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, bool fp16) const;
	uint32_t GetFlops() const;
};
class UpSampleModule : public InferenceModule {
protected:
	int stride_w;
	int stride_h; 
	bool InitDescriptors();
public:
	UpSampleModule(const XMLElement* element, Layer* l, InferenceModule* prev);
	~UpSampleModule();
	bool Forward(ForwardContext& context);
	bool Backward(FloatTensor4D& delta);
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, bool fp16) const;
	uint32_t GetFlops() const;
};


class DeconvModule : public InferenceModule {
protected:
 
public:
	DeconvModule(const XMLElement* element, Layer* l, InferenceModule* prev);
	~DeconvModule();
	bool Forward(ForwardContext& context);
	bool Backward(FloatTensor4D& delta);
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, bool fp16) const { return false; }
	uint32_t GetFlops() const { return 0; }
};