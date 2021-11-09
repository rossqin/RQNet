#pragma once
#include "layer.h"
#include "inference_module.h"
#include "param_pool.h"

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
	int layer_index;
};
struct TrainingResult { 
	float old_loss;
	float obj_loss;
	float cls_loss;
	float box_loss;

	int gt_count;		// 总共处理了多少个ground truths
	int bg_count;
	int gt_recall;		// 训练过程中对应的anchor的objectness > 0.5

	float iou;			// 对于ground_truths, 全部的iou是多少 

	float ciou;

	float object_conf;	// 和ground_truth 相关的总confidence
	float bg_conf;		// backgound confidence 
	float cls_conf;		// class confidence 

	int recalls;
	int recalls_75;
};
class InferenceModule;
typedef vector<ObjectInfo> * LPObjectInfos;
enum NMSType { DEFAULT_NMS = 0, GREEDY_NMS, DIOU_NMS, CORNERS_NMS };
 
class CNNNetwork {
protected: 
	int override_height;
	int override_width;
	int mini_batch;
	int input_channels;
	int input_width;
	int input_height;
	float* input;
	LPObjectInfos *truths;
	NMSType nms_type; 
	int detection_layers;
	vector<string> classes;
	vector< pair<float, float> > anchors;
	
	pair<int, float> classfied_result;
	vector<DetectionResult> detections;
	vector<TrainingResult> training_results;
	ModulePool module_pool;
	string def_actvation;
	cudnnTensorFormat_t data_format;
	cudnnDataType_t data_type;
	RotateType* current_training_rotates;
	int data_size; 
	bool Forward(bool training = true);
	bool Backward();
	friend class InferenceModule;

public:
	int cur_iteration; 
	vector<Layer *> layers; 
	vector<string> current_training_files; // for debugging 
	string name;
	ParamPool weights_pool;
	ParamPool adam_weights_pool;
	CNNNetwork(int h = 0, int w = 0);
	~CNNNetwork();
	inline int GetInputChannels() const { return input_channels; }
	inline int GetInputHeight() const { return input_height; }
	inline int GetInputWidth() const { return input_width; }
	inline int MiniBatch() const { return mini_batch; }
	inline void AddTrainingResult( const TrainingResult& tr) { training_results.push_back(tr); } 
	
	inline const char* Precision() const { return (data_type == CUDNN_DATA_FLOAT) ? "FP32" : "FP16"; }
	
	inline int GetAnchorCount() const { return (int)anchors.size(); } 
	inline const string& DefaultActivation() const { return def_actvation; }

	inline cudnnTensorFormat_t DataFormat() const { return data_format; }
	inline cudnnDataType_t DataType() const { return  data_type; }
	inline int DataSize() const { return data_size; }
	void AddDetectionResult(const DetectionResult& data);
	inline void SetClassfiedResult(int class_id, float confidence) { classfied_result.first = class_id, classfied_result.second = confidence; }

	inline int GetClassCount() const { return (int)classes.size(); }
	inline const string& GetClassName(int i) const { return classes[i]; }
	inline int GetDetectionLayerCount() const { return detection_layers; }
	inline int GetLayerAnchorCount() const { return (int)anchors.size() / detection_layers; }
	inline const LPObjectInfos GetBatchTruths(int b) const { if (b >= 0 && b < mini_batch) return truths[b]; return nullptr; }

	bool Load(const char* filename, cudnnDataType_t dt = CUDNN_DATA_DOUBLE);
	Layer* GetLayer(int index) const ;
	bool GetAnchor(int index, float& width, float& height, bool normalized = true);
	bool Train(bool restart);
	bool Detect(const char* path);
	bool Classify(const char* path);
	bool Eval(bool all = false);

	bool CreateOpenVINOIRv7(const string& dir, const string& ir_name, bool fp16 = true);
	void GetAnchorsStr(string& str) const ;
	bool CheckAndPrune(const char* weights_file, float c_threshold, float w_threshold=1.0e-6f);
};
