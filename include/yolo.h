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
#include "box.h"
struct ObjectInfo; 

struct TruthInLayer {
	
	int cell_x; // 本格子的 x坐标
	int cell_y; // 本个字的 y坐标
	float x_offset;
	float y_offset;
	int best_anchor_index; // in layer
	float best_anchor_width;
	float best_anchor_height;

	int best_recall_x;
	int best_recall_y;
	int best_recall_anchor;
	//int cur_pred_x;
	//int cur_pred_y;
	//int cur_anchor_index;

	int class_id;
	int recalls; 
	Box box;
	const ObjectInfo* original;  // 在标记文件中的框框，主要用来报告错误用
	float best_iou;// 可能没什么用 
};

class YoloModule : public InferenceModule {
protected:
	int cells_count;
	string mask_anchor_str;
	vector<AnchorBoxItem> masked_anchors; 
	int EntryIndex(int anchor, int loc, int entry); 
	bool Resize(int w, int h);
	bool Detect(); 
	void DeltaClass(float* output, float* delta, int cls_index, int class_id, float* p_class_conf = nullptr);
	float DeltaTruth(const TruthInLayer& truth, float* o, float* d, int cells, RotateType rt, 
		float& object_conf, float& class_conf, int& class_identified);
	bool CalcDelta();
	//bool RescueMissTruth(TruthInLayer& missT, CpuPtr<int>& truth_map, int miss_truth_index, RotateType rt);
public:
	YoloModule(const XMLElement* element, Layer* l, CNNNetwork* network, InferenceModule* prev);
	~YoloModule();
	bool Forward(ForwardContext& context); 
	bool Backward(CudaTensor& delta);
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, int& l_index) const;
	uint32_t GetFlops() const;
};