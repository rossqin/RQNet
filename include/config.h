#pragma once
#include "tinyxml2.h"
using namespace tinyxml2;  

class Dataset {
	string name;
	mutable vector<pair<string,int> > filenames;
	vector<string> classes;
public:
	Dataset(const XMLElement* element);
	~Dataset() {}
	inline size_t GetSize() const { return filenames.size(); }
	inline const string& FilenameAt(size_t i) const { return filenames[i].first; }
	inline int FileClassAt(size_t i) const { return filenames[i].second; }
	inline const string& GetName() const { return name; }
	void ShuffleFiles() const ;
};
struct AdamConfig {
	float alpha;
	float beta1;
	float beta2;
	float epsilon;
	AdamConfig() { alpha = 0.001f; beta1 = 0.9f; beta2 = 0.999f; epsilon = 1.0e-8f; }
};
struct SgdConfig {
	float base_rate;
	int burnin;
	int step;
	float power;
	float gamma; 
	float scale;
	float momentum;
	vector< pair<int, float> > steps;
	enum LearningRatePolicy {
		CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
	} policy;
	SgdConfig() {
		base_rate = 1.0e-5f; 
		burnin = 1000; 
		power = 4.0f; 
		gamma = 1.0f; 
		step = 1; 
		policy = CONSTANT; 
		scale = 1.0f; 
		momentum = 0.9f;
	}

};
enum ParamsUpdatePolicy {
	SGD = 0 ,
	Adam 
};
class AppConfig {
protected:  
	vector<Dataset*> datasets;

	int stop_interation; 

	bool save_input;
	bool neg_mining;
	bool channel_pruning;
	bool save_adam_params;

	int save_weight_interval;
	

	//da augmentation
	float da_jitter;
	float da_saturation;
	float da_exposure;
	float da_hue;

	bool fast_resize;
	//multi_scale

	bool ms_enable;
	int ms_interval;
	vector< pair<int, int> > scales;

	int batch;
	int subdivision; 

	float thresh_hold;
	float mns_thresh_hold;
	bool freeze_conv_params;
	bool freeze_bn_params;
	bool focal_loss;

	ParamsUpdatePolicy update_policy;
	AdamConfig adam_config;
	SgdConfig sgd_config;
	float decay;

	bool training_show_layers;
	bool use_ciou_loss;
	bool adversarial;

	//return dataset name
	const char* LoadTrainingSection(XMLElement* root); 
public :
	bool tracing;
	AppConfig();
	~AppConfig();
	// mode 
	// 0 - training
	// 1 - test
	// 2 - detect
	// 3 - demo 
	bool Load(const char* filename, int mode = 0);

	
	inline bool IsLastIteration(int i) const { return i >= stop_interation; } 
  
	inline int  GetLastIteration() const { return stop_interation; }
	inline bool SaveInput() const { return save_input; }
	inline bool NeedNegMining() const { return neg_mining; }

	inline float GetJitter() const { return da_jitter; }
	inline float GetSaturation() const { return da_saturation; }
	inline float GetExposure() const { return da_exposure; }
	inline float GetHue() const { return da_hue; }

	inline const AdamConfig& GetAdamConfig() const { return adam_config; }
	inline const SgdConfig& GetSgdConfig() const { return sgd_config; }
	inline float Decay() const { return decay * batch; }
	inline int GetBatch() const { return batch; }
	inline int GetSubdivision() const { return subdivision; } 

	inline bool MultiScaleEnable() const { return ms_enable; }
	inline int GetMultiScaleInterval() const { return ms_interval; } 

	inline float ThreshHold() const { return thresh_hold; }
	inline void ThreshHold(float t) { thresh_hold = t; }
	inline float NMSThreshold() const { return mns_thresh_hold; }

	inline bool ConvParamsFreezed() const { return freeze_conv_params; }
	inline bool BNParamsFreezed() const { return freeze_bn_params; }
	inline bool FocalLoss() const { return focal_loss; } 
	inline bool PruneChannels() const { return channel_pruning; }
	inline bool SaveAdamParams() const { return save_adam_params; }

	inline bool FastResize() const { return fast_resize; }
	inline int GetDatasetCount() const { return (int)datasets.size(); }

	inline ParamsUpdatePolicy UpdatePolicy() const { return update_policy; }
	const Dataset* GetDataSet(int i = 0) const;
	bool RadmonScale(uint32_t it, int& new_width, int& new_height) const;

	inline bool SaveIteration(uint32_t it) const { return (save_weight_interval > 0) && (it % save_weight_interval) == 0; }
	float GetCurrentLearningRate(int iteration) const;
	
	inline bool UseCIoULoss() const { return use_ciou_loss; }
	inline bool Adversarial() const { return adversarial; }
	
}; 
AppConfig& GetAppConfig();
