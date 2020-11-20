#pragma once
#include "inference_module.h"
#pragma pack(push)
#pragma pack(1)
/*
struct YoloTargetLabel {
	float x;
	float y;
	float w;
	float h;
	float probility;
	float class_ids[1];
};
*/
#pragma pack(pop) 
struct AnchorBoxItem {
	float width;
	float height;
	int masked_index;
};
#include "box.h"
struct ObjectInfo; 

struct TruthInLayer {
	
	int cell_x; // �����ӵ� x����
	int cell_y; // �����ӵ� y����
	float x_offset;
	float y_offset;
	float best_iou;
	int best_anchor;  //��ѵ�Ԥ���

	int class_id; 
	Box box;
	int orig_index;// �ڱ���ļ��еĿ����Ҫ������������� 
	
};

class YoloModule : public InferenceModule {
protected:
	struct AnchorLoc {	
		int x;
		int y;
		int a;
	};
	bool train_bg;
	int cells_count; 
	vector<TruthInLayer> gts; 
	string mask_anchor_str;
	vector<AnchorBoxItem> masked_anchors; 
	vector<AnchorLoc> bg_anchors;
	bool Resize(int w, int h);
	bool Detect(); 
	void DeltaClass(float* output, float* delta, int cls_index, int class_id); 
	bool CalcDelta();
	bool ResolveGTs(int batch, int& cfl_t_count);
	void CalcBgAnchors();
	int EntryIndex(int x, int y, int a, int channel);
	void UpdateDetectPerfs(float* output_cpu, vector<DetectionResult>& results);
public:
	YoloModule(const XMLElement* element, Layer* l, CNNNetwork* network, InferenceModule* prev);
	~YoloModule();
	bool Forward(ForwardContext& context); 
	bool Backward(CudaTensor& delta);
	bool OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, int& l_index) const;
	uint32_t GetFlops() const;
};
