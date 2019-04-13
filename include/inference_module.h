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
	int input_height;
	int input_width;
	int input_channels;
	int output_channels ;
	InferenceModule* logical_prev;
	cudnnTensorDescriptor_t x_desc, y_desc;
	Layer* layer;
	
	vector<InferenceModule*> prevs;
	void GetPrevModules(const XMLElement* element);	 
	virtual bool InitDescriptors(bool trainning) { return true; }
	bool PrepareShortcutDelta();
public:
	string name;
	FloatTensor4D input;
	FloatTensor4D output;
	FloatTensor4D shortcut_delta;
	static InferenceModule* FromXmlElement(const XMLElement* element,  Layer* layer, TensorOrder order, InferenceModule* prev);
	inline int GetOutputChannels() const { return output_channels; }
	bool UpdateShortcutDelta(const FloatTensor4D& delta);
	virtual bool Forward(ForwardContext& context) ;
	virtual bool Backward(FloatTensor4D& delta) ;
	virtual bool UpdateParams(float lr) { return true; }
	virtual bool DistributeDeltas(FloatTensor4D& delta);

	

	virtual ~InferenceModule() {};
};
class ConvolutionalModule : public InferenceModule {
protected:
	FloatTensor4D w;
	FloatTensor4D dw;
	//TODO: think about if not followed by batchnorm module.
	FloatTensor4D bias;
	FloatTensor4D dbias;
	cudnnTensorDescriptor_t db_desc; 

	int padding_w;
	int padding_h;
	int stride_w;
	int stride_h;
	
	cudnnFilterDescriptor_t w_desc;
	cudnnConvolutionDescriptor_t conv_desc;

	cudnnConvolutionFwdAlgo_t fwd_algo;
	cudnnConvolutionBwdDataAlgo_t bwdd_algo;
	cudnnConvolutionBwdFilterAlgo_t bwdf_algo; 
	bool InitDescriptors(bool trainning);
	friend class BatchNormModule;
public :
	~ConvolutionalModule();
	ConvolutionalModule(const XMLElement* element, Layer* l, TensorOrder order, InferenceModule* prev);
	bool Forward(ForwardContext& context);
	bool Backward(FloatTensor4D& delta);
	bool UpdateParams(float lr);
	
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
	bool InitDescriptors(bool trainning);
public :
	PoolingModule(const XMLElement* element, Layer* l, TensorOrder order, InferenceModule* prev);
	~PoolingModule();
	bool Forward(ForwardContext& context);
	bool Backward(FloatTensor4D& delta);
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
	bool InitDescriptors(bool trainning);
public:
	BatchNormModule(const XMLElement* element, Layer* l, TensorOrder order, InferenceModule* prev);
	~BatchNormModule();	
	bool Forward(ForwardContext& context);
	bool Backward(FloatTensor4D& delta);
	bool UpdateParams(float lr);

};
class ActivationModule : public InferenceModule {
protected:
	ACTIVATION_TYPE atype;
	float factor; 
public:
	ActivationModule(const XMLElement* element, Layer* l, TensorOrder order, InferenceModule* prev);
	~ActivationModule(){ }
	bool Forward(ForwardContext& context);
	bool Backward(FloatTensor4D& delta);
};
class UpSampleModule : public InferenceModule {
protected:
	int stride_w;
	int stride_h; 
	bool InitDescriptors(bool trainning);
public:
	UpSampleModule(const XMLElement* element, Layer* l, TensorOrder order, InferenceModule* prev);
	~UpSampleModule();
	bool Forward(ForwardContext& context);
	bool Backward(FloatTensor4D& delta);
};


class DeconvModule : public InferenceModule {
protected:
 
public:
	DeconvModule(const XMLElement* element, Layer* l, TensorOrder order, InferenceModule* prev);
	~DeconvModule();
	bool Forward(ForwardContext& context);
	bool Backward(FloatTensor4D& delta);
};