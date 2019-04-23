#include "stdafx.h"
#include "network.h"
#include "param_pool.h"
#include "yolo.h"
#include "box.h"
#include "config.h"
#include <memory>
YoloModule::YoloModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev): 
	InferenceModule(element, l,net, prev) {
	 
	GetPrevModules(element);
	focal_loss = element->BoolAttribute("focal-loss", false);

	const char* s = element->Attribute("anchor-masks");
	if (s)
		mask_anchor_str = s;
	else
		mask_anchor_str = ""; 
	vector<string> strs;
	split_string(strs, mask_anchor_str);
	AnchorBoxItem abi;
	for (string& s : strs) {
		abi.masked_index = atoi(s.c_str()); 
		if (network->GetAnchor(abi.masked_index, abi.width, abi.height)) { 
			masked_anchors.push_back(abi);
		}
		
	}
	

	features = input_channels / (int)masked_anchors.size();
	classes = features - 5;
	if (classes < 0) { // exceptional situation
		features = 6;
		classes = 1;
	} 
	output.DataType(CUDNN_DATA_FLOAT);
	Resize(input_width, input_width);
	output_channels = input_channels;
	object_loss_factor = 5.0f;
	class_loss_factor = 2.0f;

}
YoloModule::~YoloModule() { 
}
enum {
	INDEX_PROBILITY_X = 0,
	INDEX_PROBILITY_Y,  // 1
	INDEX_PROBILITY_W,  // 2
	INDEX_PROBILITY_H,  // 3
	INDEX_CONFIDENCE,   // 4
	INDEX_PROBILITY_CLASS_0,
	INDEX_PROBILITY_CLASS_1,
	INDEX_PROBILITY_CLASS_2,
	INDEX_PROBILITY_CLASS_3
	//...
};
static Box get_yolo_box(const float *data, const AnchorBoxItem& anchor, int x, int y, float reciprocal_w, float reciprocal_h, int stride) {
	Box b;
	b.x = (*data + x) * reciprocal_w;// reciprocal_w == 1 / w 
	data += stride;
	b.y = (*data + y) * reciprocal_h;// reciprocal_h == 1 / h 
	data += stride;
	if (*data < 10.0) // to avoid overflow
		b.w = exp(*data) * anchor.width;
	else
		b.w = 0.00001f;

	data += stride;
	if (*data < 10.0)
		b.h = exp(*data) * anchor.height;
	else
		b.h = 0.00001f;
	return b;
}
 
int YoloModule::EntryIndex(int anchor, int loc, int entry) {
 
	return (anchor * features + entry) * output.Elements2D() + loc;
}
void YoloModule::DeltaBackground(float* data, float* delta, const vector<ObjectInfo>& truths, float& avg_anyobj) {
	float reciprocal_w = 1.0 / output.Width();
	float reciprocal_h = 1.0 / output.Height(); 
	for (int y = 0, offset = 0; y < output.Height(); y++) {
		for (int x = 0; x < output.Width(); x++, offset++) {
			for (int n = 0; n < (int) masked_anchors.size(); n++) { // l.n == 3
															  // sigmoid(tx),sigmoid(ty),tw,th,Pr(Obj),Pr(Cls_0|Obj),Pr(Cls_0|Obj)...
				int box_index = EntryIndex(n, offset, INDEX_PROBILITY_X); 
				Box pred = get_yolo_box(data + box_index, masked_anchors[n],
					x, y, reciprocal_w, reciprocal_h, output.Elements2D());
				float best_iou = 0;
				int best_t = 0;
				for (int t = 0; t < (int)truths.size(); t++) {
					const ObjectInfo& truth = truths[t];
					if (truth.class_id < 0) break;
					float iou = BoxIoU(pred, Box(truth));
					if (iou > best_iou) { best_iou = iou; best_t = t; }
				}
				// class_index : point to Pr(Obj)
				int object_index = box_index + (INDEX_CONFIDENCE - INDEX_PROBILITY_X) * output.Elements2D();
				avg_anyobj += data[object_index]; 
				if (best_iou > 0.7) {					
					delta[object_index] = 0.0f;//初始化全是背景，离目标较近的忽略，梯度为0
				} 
				else {
					delta[object_index] = 0.0f - data[object_index];
				}

				// 更近的视为目标(不过这一步效果不太好，所以阈值设为1，永远不执行)
				if (best_iou > 1.0) {
					delta[object_index] = 1 - data[object_index];
					const ObjectInfo& truth = truths[best_t];//network->Truth(b, best_t);
					int class_id = truth.class_id;
					int class_index = object_index + output.Elements2D(); // * (INDEX_PROBILITY_CLASS_0 - INDEX_CONFIDENCE) ;
					DeltaClass(data, delta, class_id, class_index);
					DeltaBox(data,delta,truth, masked_anchors[n], box_index, x, y);

				}
			} // masked_anchor
		} // x
	}// y 
}
void YoloModule::DeltaBox(float* data, float* delta, const ObjectInfo& truth, const AnchorBoxItem& anchor, int index, int x, int y) {

	float tx = (truth.x * output.Width() - x);
	float ty = (truth.y * output.Height() - y);
	float tw = log(truth.w / anchor.width);
	float th = log(truth.h / anchor.height);
	float scale = (2.0f - truth.w * truth.h);
	int size = output.Elements2D();

	delta[index] = scale * (tx - data[index]);
	index += size;
	delta[index] = scale * (ty - data[index]);
	index += size;
	delta[index] = scale * (tw - data[index]);
	index += size;
	delta[index] = scale * (th - data[index]);
}
bool YoloModule::Resize(int w, int h) {
	input_height = h;
	input_width = w;
	output_width = input_width;
	output_height = input_height; 
	return true;
}
 
bool YoloModule::Detection() {
	std::cout << "Start Analyze the " << input_width << "x" << input_height << " layer ...\n";
	CpuPtr<float> output_cpu(output.Elements3D(), output.Data());
	if (output_cpu.GetError()) {
		cerr << " Error: Copy memory failed in " << name << "!\n";
		return false;
	}
	DetectionResult dr ;
	float reciprocal_w = 1.0f / input_width;
	float reciprocal_h = 1.0f / input_height;
	int size = output.Elements2D();
	float threshold = GetAppConfig().ThreshHold();
	for (int y = 0,i = 0; y < output.Height(); y++) {
		for (int x = 0; x < output.Width(); x++, i++) {
			int best_n = -1;
			dr.confidence = -9999.0;
			for (int n = 0; n < (int)masked_anchors.size(); n++) {
				int box_index = EntryIndex(n, i, INDEX_PROBILITY_X);
				int conf_index = box_index + (INDEX_CONFIDENCE - INDEX_PROBILITY_X) * size;
				float confidence = output_cpu[conf_index];
				if (confidence < threshold) {
					continue;
				}				
				
				if (confidence < dr.confidence) {
					std::cout << "Not of the best confidence, ignore...\n";
					continue;
				}
				int class_id = -1;
				float best_cls_conf = -9999.0;
				int cls_index = conf_index + size;
				for (int c = 0; c < classes; c++, cls_index += size) {
					if (output_cpu[cls_index] > threshold && 
						output_cpu[cls_index] > best_cls_conf) {
						class_id = c;
						best_cls_conf = output_cpu[cls_index];
					}
				}
				if (-1 == class_id) {
					std::cout << "Class confidence below threshold, ignore...\n";
					continue;
				}				
				best_n = n;
				dr.class_id = class_id;
				dr.class_confidence = best_cls_conf;
				dr.confidence = confidence;
				Box box =  get_yolo_box(output_cpu.ptr + box_index, masked_anchors[n],
					x, y, reciprocal_w, reciprocal_h, size);
				std::cout << " Found at grid [" << x << "," << y << "], anchor: " << n << ", prob: " << confidence
					<<", position: (" << (int)(box.x * 416.0f)<< ","<< (int)(box.y * 416.0f) << ")\n";
				dr.x = box.x;
				dr.y = box.y;
				dr.w = box.w;
				dr.h = box.h; 
			}
			if (best_n > 0) { 
				network->AddDetectionResult(dr);
			}
		}
	}
	return true;
}
void YoloModule::DeltaClass(float* data, float* delta, int class_id, int index, float* avg_cat) {

	int ti = index;
	if (class_id > 0)  ti += output.Elements2D() * class_id;
	if (delta[ti]) { //TOCHECK: always false?
		delta[ti] = 1 - data[ti];
		if(avg_cat) *avg_cat += data[ti];
		return;
	}
	// Focal loss
	if (focal_loss) {
		// Focal Loss
		float alpha = 0.5f;    // 0.25 or 0.5
							   //float gamma = 2;    // hardcoded in many places of the grad-formula


		float grad = 1.0f, pt = data[ti];
		// http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d

		// http://blog.csdn.net/linmingan/article/details/77885832
		if (pt > 0.0f) grad = (pt - 1.0f) * (2.0f * pt * logf(pt) + pt - 1.0f);

		grad *= alpha;
		for (int n = 0; n < classes; n++, index += output.Elements2D()) {
			delta[index] = (((n == class_id) ? 1.0f : 0.0f) - data[index]);
			delta[index] *= grad;
			if (n == class_id) {
				if(avg_cat) *avg_cat += data[index];
			}
		}
	}
	else {
		// default
		for (int n = 0; n < classes; n++, index += output.Elements2D()) {
			delta[index] = ((n == class_id) ? 1 : 0) - data[index];
			if (n == class_id && avg_cat) {
				*avg_cat += data[index];
			}
		}
	}
}
const float one = 1.0, zero = 0.0;
bool YoloModule::Forward(ForwardContext& context) {
	if (!InferenceModule::Forward(context)) return false;
	if (input.DataFormat() != CUDNN_TENSOR_NCHW) return false;
	int b, n;

	if (input.DataType() == CUDNN_DATA_FLOAT) {
		output = input;
	}
	else {
		float *out = reinterpret_cast<float *>(output.Data());
		half *in = reinterpret_cast<__half*>(input.Data());
		if(!f16_to_f32( out, in, input.Elements())) return false;
	}

	float* output_gpu = reinterpret_cast<float*>(output.Data());
 
	bool ret = true; 
	int total_cells = output_height * output_width;
	int xy_elements = 2 * total_cells;
	int conf_cls_elements = (1 + classes) * total_cells;
	for (b = 0; b < output.Batch(); b++) {
		for (n = 0; n < (int)masked_anchors.size(); n++) {
			float* x = output_gpu + EntryIndex(n, 0, INDEX_PROBILITY_X);
			if (!activate_array_ongpu(x,x, xy_elements, output.DataType(), LOGISTIC)) {
				return false;
			} 
			x += (INDEX_CONFIDENCE - INDEX_PROBILITY_X) * total_cells;
			if (!activate_array_ongpu(x, x, conf_cls_elements, output.DataType(), LOGISTIC))
				return false; 
		}
		output_gpu += total_cells * output_channels;
	} 
	if (!context.training) {   
		return Detection();
	}

	if (shortcut_delta.SameShape(input))
		shortcut_delta = 0.0f;
	else if (!shortcut_delta.Init(input.Batch(), input.Channel(), input.Height(), input.Width()))
		return false;
	float loss = 0.0;
	float avg_iou = 0, recall = 0, recall75 = 0, avg_cat = 0;
	float avg_obj = 0, avg_anyobj = 0;
	int count = 0, class_count = 0;
	float reciprocal_w = 1.0f / input.Width();
	float reciprocal_h = 1.0f / input.Height();
	CpuPtr<float> delta_cpu(output.Elements3D());
	CpuPtr<float> output_cpu(output.Elements3D());

	size_t batch_bytes = output.Elements3D() * sizeof(float);
	int size = output.Elements2D();
	for (int b = 0; b < output.Batch(); b++) {
		const LPObjectInfos truths = network->GetBatchTruths(b);
		if (!truths) return false;
		float* output_gpu = reinterpret_cast<float*>(output.BatchData(b));
		if (cudaSuccess != cudaMemcpy(output_cpu, output_gpu, batch_bytes, cudaMemcpyDeviceToHost)) {
			return false;
		}
		delta_cpu.Reset();
		DeltaBackground(output_cpu, delta_cpu, *truths, avg_anyobj);
		for (int t = 0; t < (int)truths->size(); t++) {
			const ObjectInfo& truth = truths->at(t);
			if (truth.class_id < 0) break;
			float best_iou = 0;
			int n, best_n = 0, mask_n = -1;
			int x = (truth.x * output.Width());
			int y = (truth.y * output.Height());

			Box truth_shift(0, 0, truth.w, truth.h);
			float w, h;
			for (n = 0; n < network->GetAnchorCount(); n++) {
				network->GetAnchor(n, w, h);
				Box pred(0, 0, w, h);
				float iou = BoxIoU(pred, truth_shift);
				if (iou > best_iou) {
					best_iou = iou;
					best_n = n;
				}
			}
			for (n = 0; n < masked_anchors.size(); n++) {
				if (best_n == masked_anchors[n].masked_index) {
					mask_n = n;
					break;
				}
			}
			if (mask_n >= 0) { // found matched anchor
				int offset = y * output.Width() + x;
				int box_index = EntryIndex(mask_n, offset, INDEX_PROBILITY_X);
				DeltaBox(output_cpu, delta_cpu, truth, masked_anchors[mask_n], box_index, x, y);

				int object_index = box_index + (INDEX_CONFIDENCE - INDEX_PROBILITY_X) * size;
				avg_obj += output_cpu[object_index];
				delta_cpu[object_index] = 1 - output_cpu[object_index];

				int class_id = truth.class_id;//context.truths[t*(4 + 1) + b*l.truths + 4];

				int class_index = object_index + size /* *(INDEX_PROBILITY_CLASS_0 - INDEX_CONFIDENCE) */;
				DeltaClass(output_cpu, delta_cpu, class_id, class_index, &avg_cat);

				count++;
				class_count++;

				Box pred = get_yolo_box(output_cpu + box_index, masked_anchors[n],
					x, y, reciprocal_w, reciprocal_h, size);
				float iou = BoxIoU(pred, Box(truth));

				if (iou > 0.5f) recall += 1.0;
				if (iou > 0.75f) recall75 += 1.0;
				avg_iou += iou;
			}
		}
		if (shortcut_delta.DataType() == CUDNN_DATA_FLOAT) {
			float* delta_gpu = reinterpret_cast<float*>(shortcut_delta.BatchData(b));
			if (cudaSuccess != cudaMemcpy(delta_gpu, delta_cpu, batch_bytes, cudaMemcpyHostToDevice))
				return false;
		}
		else {
			__half* delta_gpu = reinterpret_cast<__half*>(shortcut_delta.BatchData(b));
			CudaPtr<float> ptr(output.Elements3D(), delta_cpu);
			if (!f32_to_f16(delta_gpu, ptr, output.Elements3D()))
				return false;
		}
		loss += square_sum_array(delta_cpu, output.Elements3D());

	}
	loss /= network->MiniBatch();
	network->RegisterTrainingResults(loss, 0, 0);



	ostringstream oss;
	avg_anyobj /= output.Elements();
	oss << " " << name << ": Found " << count << " objects with layer loss: " << loss;
	if (count > 0) {
		avg_iou /= count;
		avg_obj /= count;
		recall /= count;
		recall75 /= count;
		recall *= 100;
		recall75 *= 100;

		if (class_count > 0) {
			avg_cat /= class_count;
			oss << setprecision(4) << ".\n               IoU:" << avg_iou << ", Cls:" << avg_cat << ", Obj:" << avg_obj
				<< ", No Obj:" << avg_anyobj << ", .5R:" << recall << "%, .75R:" << recall75 << "%.";
		}
		else {
			oss << setprecision(4) << ".\n               IoU:" << avg_iou << ", Obj:" << avg_obj
				<< ", No Obj:" << avg_anyobj << ", .5R:" << recall << "%, .75R:" << recall75 << "%.";


		}
	}

	std::cout << oss.str() << endl << endl;

	return true;
}
bool YoloModule::Backward(CudaTensor& delta) {
	delta = shortcut_delta; 
	return shortcut_delta.Release();
}

bool YoloModule::OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, int& l_index) const {
	if (!InferenceModule::OutputIRModel(xml, bin, edges, bin_offset, l_index)) return false;
	xml << "    <layer id=\"" << index << "\" name=\"" << name << "\" precision=\"" << Precision() << "\" type=\"RegionYolo\">" << endl;
	//<data />
	//<data axis="1" classes="1" coords="4" do_softmax="0" end_axis="3" mask="0,1,2" num="9"/>
	string str;
	network->GetAnchorsStr(str);
	xml << "      <data anchors=\""<< str<< "\" axis=\"1\" classes=\"" << classes << "\" coords=\"4\"  do_softmax=\"0\" end_axis=\"3\" mask=\"" <<  
		mask_anchor_str <<"\" num=\""<< network->GetAnchorCount() <<"\" />" << endl;
	WritePorts(xml);
	xml << "    </layer>" << endl;
	return true;
}
uint32_t YoloModule::GetFlops() const {
	return 0;
}