#pragma once
#include "tinyxml2.h"
using namespace tinyxml2; 
enum LearningRatePolicy {
	CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
};
class Dataset {
	mutable vector<string> filenames;
	vector<string> classes;
public:
	Dataset(const XMLElement* element);
	~Dataset() {}
	inline size_t GetSize() const { return filenames.size(); }
	inline const string& FilenameAt(size_t i) const { return filenames[i]; }
	inline size_t GetClasses() const { return classes.size(); }
	inline const string& ClassAt(size_t i) const { return classes[i]; }
	void ShuffleFiles() const ;
};

class AppConfig {
protected:  
	Dataset* dataset;
	int stop_interation;
	bool restart_interation;

	bool save_input;
	string input_dir;

	int save_weight_interval;
	string weight_file_prefix;
	string out_dir;
	float momentum;
	float decay;

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

	//learning_rates ;
	float lr_base;
	int lr_burnin;
	int lr_step;
	float lr_scale;
	float lr_power;
	float lr_gamma;
	LearningRatePolicy lr_policy;

	vector< pair<int, float> > lr_steps;
	vector<string> classes;

	float detect_threshhold;

	int max_truths;

	bool freezeConvParams;
	bool freezeBNParams;
	bool freezeActParams;
	bool small_object;

	string update_strategy;

	//return dataset name
	const char* LoadTrainingSection(XMLElement* root);
	const char* LoadTestingSection(XMLElement* root);
	void LoadDetectSection(XMLElement* root);
public :
	
	AppConfig();
	~AppConfig();
	// mode 
	// 0 - training
	// 1 - test
	// 2 - detect
	// 3 - demo 
	bool Load(const char* filename, int mode = 0);

	inline const Dataset* GetDataSet() const { return dataset; }	
	inline bool IsLastIteration(int i) const { return i >= stop_interation; 	}
	inline bool FromFirstIteration() const { return restart_interation; }
  
	inline int  GetLastIteration() const { return stop_interation; }
	inline bool SaveInput() const { return save_input; }
	inline const string& SaveInputDir() const { return input_dir; }


	inline float GetJitter() const { return da_jitter; }
	inline float GetSaturation() const { return da_saturation; }
	inline float GetExposure() const { return da_exposure; }
	inline float GetHue() const { return da_hue; }


	inline int GetBatch() const { return batch; }
	inline int GetSubdivision() const { return subdivision; } 

	inline bool MultiScaleEnable() const { return ms_enable; }
	inline int GetMultiScaleInterval() const { return ms_interval; }
	inline int GetMaxTruths() const { return max_truths; }

	bool GetClass(int i, string& result) const;
	inline int GetClasses() const { return (int)classes.size(); }
	inline float GetThreshhold() const { return detect_threshhold; }

	inline bool ConvParamsFreezed() const { return freezeConvParams; }
	inline bool BNParamsFreezed() const { return freezeBNParams; }
	inline bool ActParamsFreezed() const { return freezeActParams; }
	inline bool SmallObjEnabled() const { return small_object; } 

	inline bool FastResize() const { return fast_resize; }
	inline float Decay()const { return decay; }
	inline float Momentum() const { return momentum; }

	inline const string& UpdateStrategy() const { return update_strategy; }

	bool RadmonScale(uint32_t it, int& new_width, int& new_height) const;
	
	bool GetWeightsPath(uint32_t it, string& filename) const ; 
	float GetCurrentLearningRate(int iteration) const;
	

	
}; 
AppConfig& GetAppConfig();
