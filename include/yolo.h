#pragma once
#include "inference_module.h"
#pragma pack(push)
#pragma pack(1)
struct YoloTargetLabel {
	float x;
	float y;
	float w;
	float h;
	float probility;
	float class_ids[1];
};

#pragma pack(pop) 
struct AnchorBoxItem {
	float width;
	float height;
	int masked_index;
};

struct ObjectInfo;
class YoloModule : public InferenceModule {
protected:
	int features;
	int classes;
	bool focal_loss; 
	float threshold_ignore;
	float threshold_thruth; 
	vector<AnchorBoxItem> masked_anchors;
	int EntryIndex(int anchor, int loc, int entry);
	void DeltaBackground(float* data, float* delta, ObjectInfo* truths, int max_boxes, float& avg_anyobj);
	void DeltaClass(float* data, float* delta, int class_id, int index, float* avg_cat = NULL);
	void DeltaBox(float* data, float* delta, const ObjectInfo& truth, const AnchorBoxItem& anchor, int index, int x, int y);
	bool InitDescriptors(bool trainning);
public:
	YoloModule(const XMLElement* element, Layer* l, TensorOrder order, InferenceModule* prev);
	~YoloModule();
	bool Forward(ForwardContext& context);
	bool Backward(FloatTensor4D& delta);
};