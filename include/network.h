#pragma once
#include "layer.h"
#include "inference_module.h"
#pragma pack(push)
#pragma pack(1)
struct ObjectInfo {
	float x;
	float y;
	float w;
	float h;
	float class_id;
};
#pragma pack(pop) 
struct DetectionResult {
	int class_id;
	float x;
	float y;
	float w;
	float h;
	float class_confidence;
	float confidence;
};
class InferenceModule;
typedef vector<ObjectInfo> * LPObjectInfos;
class CNNNetwork {
protected:
	int mini_batch;
	int input_channels;
	int input_width;
	int input_height;
	float* input;
	LPObjectInfos *truths;
	float loss;
	float true_positives;
	float false_negatives;
	int training_batch;
	vector< pair<float, float> > anchors;
	vector<Layer *> layers;
	vector<DetectionResult> detections;
	ModulePool module_pool;
	string def_actvation;
	cudnnTensorFormat_t data_format;
	cudnnDataType_t data_type;
	int data_size;
	friend ModulePool& GetModulePool();
	bool Forward(bool training = true);
	bool Backward();
	friend class InferenceModule;
public:
	size_t workspace_size;
	void* workspace; 
	
	CNNNetwork();
	~CNNNetwork();
	inline int GetInputChannels() const { return input_channels; }
	inline int GetInputHeight() const { return input_height; }
	inline int GetInputWidth() const { return input_width; }
	inline int MiniBatch() const { return mini_batch; }
	inline void RegisterTrainingResults(  float l, float tp, float fn) {
		training_batch ++;
		loss += l;
		true_positives += tp;
		false_negatives += fn;
	}
	inline float GetLoss() const { return loss; } 

	inline const char* Precision() const { return (data_type == CUDNN_DATA_FLOAT) ? "FP32" : "FP16"; }
	
	inline int GetAnchorCount() const { return (int)anchors.size(); } 
	inline const string& DefaultActivation() const { return def_actvation; }

	inline cudnnTensorFormat_t DataFormat() const { return data_format; }
	inline cudnnDataType_t DataType() const { return  data_type; }
	inline int DataSize() const { return data_size; }
	void AddDetectionResult(const DetectionResult& data);

	inline const LPObjectInfos GetBatchTruths(int b) const { if (b >= 0 && b < mini_batch) return truths[b]; return nullptr; }

	bool Load(const char* filename, cudnnDataType_t dt = CUDNN_DATA_DOUBLE);
	Layer* GetLayer(int index) const ;
	bool GetAnchor(int index, float& width, float& height);
	bool UpdateWorkspace(size_t new_size); 	
	bool Train();
	bool Detect(const char* filename);	
	
	bool OutputIRModel(const string& dir, const string& name, int& l_index ) const;
	void GetAnchorsStr(string& str) const ;
};
CNNNetwork& GetNetwork();
ModulePool& GetModulePool();
