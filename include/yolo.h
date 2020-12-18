#pragma once
#include "inference_module.h"
 
struct AnchorBoxItem {
	float width;
	float height;
	int masked_index;
};
#include "box.h"
struct ObjectInfo; 

struct TruthInLayer {
	
	int cell_x; // 本格子的 x坐标
	int cell_y; // 本格子的 y坐标
	float x_offset;
	float y_offset;
	float best_iou;
	int best_anchor;  //最佳的预测框

	int class_id; 
	Box box;
	int orig_index;// 在标记文件中的框框，主要用来报告错误用 
	
};

class YoloModule : public InferenceModule {
protected:
	struct AnchorLoc {	
		int x;
		int y;
		int a;
	}; 
	float nms_beta;
	float cls_normalizer;
	float iou_normalizer;
	float obj_normalizer;
	float bg_normalizer;
	float ignore_thresh;
	float truth_thresh;
	bool objectness_smooth;
	float max_delta;
	bool train_bg;
	int cells_count; 
	vector<TruthInLayer> gts; 
	string mask_anchor_str;
	vector<AnchorBoxItem> masked_anchors; 
	vector<AnchorLoc> bg_anchors;
	bool Resize(int w, int h);
	bool Detect(); 
	void DeltaClass(float* o, float* d, int cls_index, int class_id);
	float DeltaBox(float* o, float* d, const TruthInLayer& gt, Box pred, float anchor_w, float anchor_h);
	bool CalcDelta();
	bool ResolveGTs(int batch, int& cfl_t_count, int& gt_count);
	void CalcBgAnchors(int& bg_count);
	int EntryIndex(int x, int y, int a, int channel);
	void UpdateDetectPerfs(float* output_cpu, vector<DetectionResult>& results);
public:
	inline const char* GetMaskedAnchorString() const { return mask_anchor_str.c_str(); }
	YoloModule(const XMLElement* element, Layer* l, CNNNetwork* network, InferenceModule* prev); 
	bool Forward(ForwardContext& context); 
	bool Backward(CudaTensor& delta);
	bool RenderOpenVINOIR(vector<OpenVINOIRv7Layer>& layers, vector<OpenVINOIRv7Edge>& edges, ofstream& bin, size_t& bin_offset, bool fp16)  {
		return true;
	}
	void WriteOpenVINOOutput(ofstream& xml) const;
	uint32_t GetFlops() const;
};
