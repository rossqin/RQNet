#pragma once
#include "tensor.h"
#include "tinyxml2.h"
using namespace tinyxml2;
class InferenceModule;
struct ObjectInfo;
struct ForwardContext {
	bool training;
	bool freezeConvParams;
	bool freezeBNParams;
	bool freezeActParams;  
	FloatTensor4D& input;
	int max_truths_per_batch;
	ObjectInfo* truths;

};
class Layer {
protected:
	int index;
	string name;
	vector<InferenceModule*> modules;

public:
	InferenceModule* last_module;
	Layer(const XMLElement* element, int i);
	inline int GetIndex() const { return index; }
	inline const string& GetName() const { return name; }


	~Layer() {}
	bool Forward(ForwardContext& context)  ;
	bool Backward(FloatTensor4D& delta) ;

	bool Update(float lr);
 
	void Print() const ;
	bool LoadWeights(ifstream& f, bool transpose) { return true; }
	bool SaveWeigths(ofstream& f) { return true; }
	bool GetCost(float& val) const { return false; }
	int  MemRequired() const { return 0; }
	unsigned int GetWorkspaceSize() const { return 0; }

};
