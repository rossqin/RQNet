#pragma once
#include "tinyxml2.h"
using namespace tinyxml2;
class InferenceModule;
struct ObjectInfo;
class CudaTensor;
class CNNNetwork;
struct ForwardContext {
	bool training;
	bool freezeConvParams;
	bool freezeBNParams; 
	CudaTensor* input;

};
class Layer {
protected:
	int index;
	string name;
	string desc;
	vector<InferenceModule*> modules;
	CNNNetwork* network;
	friend class InferenceModule;
public:
	friend class CNNNetwork;
	// for debugging
	
	Layer(const XMLElement* element, CNNNetwork* net, InferenceModule*& prev_module);
	inline int GetIndex() const { return index; }
	inline const string& GetName() const { return name; }
	inline const string& GetDesc() const { return desc; }
	inline InferenceModule* GetFirstModule() const { return (modules.size() > 0) ? modules.front() : nullptr; }
	~Layer() {}
	bool Forward(ForwardContext& context)  ;
	bool Backward(CudaTensor& delta) ;

	bool FuseBatchNormModule();

	bool Update(float lr);
 
	void Print() const ;
	bool LoadWeights(ifstream& f, bool transpose) { return true; }
	bool SaveWeigths(ofstream& f) { return true; }
	bool GetCost(float& val) const { return false; }
	int  MemRequired() const { return 0; }
	unsigned int GetWorkspaceSize() const { return 0; }
	
};
