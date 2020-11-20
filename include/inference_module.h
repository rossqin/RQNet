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
class ShortcutModule;
struct PerfData {
	int gt_count;		// �ܹ������˶��ٸ�ground truths
	int bg_count;
	int gt_recall;		// ѵ�������ж�Ӧ��anchor��objectness > 0.5
	float iou;			// ����ground_truths, ȫ����iou�Ƕ���
	float object_conf;	// ��ground_truth ��ص���confidence
	float bg_conf;		// backgound confidence 
	float cls_conf;		// class confidence
	float loss;

	int tp_50;			// iou�ı�׼Ϊ50%������µ�ture positive����
	int fp_50;			// iou�ı�׼Ϊ50%������µ�false positive����
	int tp_75;			// iou�ı�׼Ϊ75%������µ�ture positive����
	int fp_75;			// iou�ı�׼Ϊ50%������µ�false positive����
};
class InferenceModule {
protected:
	mutable int index;
	int output_port;
	int input_height;
	int input_width;
	int output_height;
	int output_width;
	int input_channels;
	int output_channels; 
	PerfData perf_data;
	 
	InferenceModule* logical_prev; 
	Layer* layer;
	CNNNetwork* network;
	
	vector<InferenceModule*> prevs;
	virtual void GetPrevModules(const XMLElement* element, bool add_channels = true);	 
	virtual bool Resize(int w, int h) = 0;
	bool PrepareShortcutDelta();
	void WritePorts(ofstream& xml) const;
	char* cached_input ;
	char* cached_output;
	static InferenceModule* last_parsing;
	friend class ShortcutModule;
public:
	string name;
	CudaTensor input, output;
	CudaTensor shortcut_delta;
	InferenceModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev);
	virtual ~InferenceModule();
	static InferenceModule* FromXmlElement(const XMLElement* element,  Layer* layer, CNNNetwork* network, InferenceModule* prev);
	inline const char* Precision() const { return (output.DataType() == CUDNN_DATA_FLOAT) ? "FP32" : "FP16"; }
	inline int GetOutputChannels() const { return output_channels; }
	bool UpdateShortcutDelta(const CudaTensor& delta);
	virtual bool Forward(ForwardContext& context) ;
	virtual bool Backward(CudaTensor& delta) ;
	virtual bool UpdateParams(float lr) { return true; }
	virtual bool DistributeDeltas(CudaTensor& delta);
	virtual bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, int &l_index) const;
	virtual uint32_t GetFlops() const = 0;
	bool CacheOutput();
};
class BatchNormModule;
class ConvolutionalModule : public InferenceModule {
protected:
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
	friend class BatchNormModule;

	int padding_w;
	int padding_h;
	int stride_w;
	int stride_h;
	int dilation_w;
	int dilation_h;
	int groups;

	size_t workspace_size;
	
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
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, int &l_index) const;
	uint32_t GetFlops() const;
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
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, int &l_index) const;
	uint32_t GetFlops() const;
};
class BatchNormModule : public InferenceModule {
protected:
	bool fused;
	CudaTensor params; 
	CudaTensor training_params;

	CudaTensor adam_params; 
	//CudaTensor adam_gamma_m;
	//CudaTensor adam_gamma_v;
	//CudaTensor adam_beta_m;
	//CudaTensor adam_beta_v;

	bool freezed;
	cudnnTensorDescriptor_t t_desc; 
	CudaTensor* forward_input;
	bool Resize(int w, int h);
public:

	BatchNormModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev);
	~BatchNormModule();	
	bool Forward(ForwardContext& context);
	bool Backward(CudaTensor& delta);
	bool UpdateParams(float lr);
	inline bool IsFused() const { return fused; }
	//Do nothing.
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, int &l_index) const { return true; }
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
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, int &l_index) const;
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
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, int &l_index) const;
	uint32_t GetFlops() const;
};


class DeconvModule : public InferenceModule {
protected:
 
public:
	DeconvModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev);
	~DeconvModule();
	bool Forward(ForwardContext& context);
	bool Backward(CudaTensor& delta);
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, int &l_index) const { return false; }
	uint32_t GetFlops() const { return 0; }
};
class ShortcutModule : public InferenceModule {
protected:
	bool Resize(int w, int h);
public:
	ShortcutModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev);
	~ShortcutModule();
	bool Forward(ForwardContext& context);
	bool Backward(CudaTensor& delta);
	virtual void GetPrevModules(const XMLElement* element, bool add_channels = true);
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, int &l_index) const;
	uint32_t GetFlops() const { return 0; }
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
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, int &l_index) const;
	uint32_t GetFlops() const { return 0; }
};