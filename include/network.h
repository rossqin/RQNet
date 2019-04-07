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
 
class CNNNetwork {
protected:
	TensorOrder data_order;
	DataType data_type;	
	float loss;
	vector< pair<float, float> > anchors;
	vector<Layer *> layers;
	ModulePool module_pool;
	string def_actvation;
	ObjectInfo* truths; 
	friend ModulePool& GetModulePool();
	bool Forward(bool training = true);
	bool Backward();
public:
	size_t workspace_size;
	void* workspace;
	FloatTensor4D input;
	
	CNNNetwork();
	~CNNNetwork();
	
	inline void RegisterLoss(float l) { loss += l; }
	inline float GetLoss() const { return loss; } 
	inline int GetAnchorCount() const { return (int)anchors.size(); }
	inline TensorOrder GetDataOrder() const { return data_order; } 
	inline const string& DefaultActivation() const { return def_actvation; }

	bool Load(const char* filename);
	Layer* GetLayer(int index) const ;
	bool GetAnchor(int index, float& width, float& height);
	bool UpdateWorkspace(size_t new_size); 
	
	bool Train();

};
CNNNetwork& GetNetwork();
ModulePool& GetModulePool();
