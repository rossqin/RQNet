#include "stdafx.h"
#include "network.h"
#include "param_pool.h"
#include "yolo.h"
#include "box.h"
YoloModule::YoloModule(const XMLElement* element, Layer* l, TensorOrder order) {
	layer = l;
	x_desc = NULL;
	y_desc = NULL;
	output_cpu = NULL;
	delta_cpu = NULL;
	input_width = 0;
	input_height = 0;
	GetPrevModules(element);
	threshold_ignore = element->FloatAttribute("ignore-thresh",0.5);
	threshold_thruth = element->FloatAttribute("truth-thresh", 1.0);
	focal_loss = element->BoolAttribute("focal-loss", false);

	const char* s = element->Attribute("anchor-masks");
	string anch_str(s ? s : "");
	
	vector<string> strs;
	split_string(strs, anch_str);
	AnchorBoxItem abi;
	for (string& s : strs) {
		abi.masked_index = atoi(s.c_str()); 
		if (GetNetwork().GetAnchor(abi.masked_index, abi.width, abi.height)) { 
			masked_anchors.push_back(abi);
		}
		
	}
	features = input_channels / masked_anchors.size();
	classes = features - 5;
	if (classes < 0) { // exceptional situation
		features = 6;
		classes = 1;
	}
}
YoloModule::~YoloModule() {
	if (output_cpu) delete[]output_cpu;
	if (delta_cpu) delete[]delta_cpu;
}
enum {
	INDEX_PROBILITY_X = 0,
	INDEX_PROBILITY_Y,  // 1
	INDEX_PROBILITY_W,  // 2
	INDEX_PROBILITY_H,  // 3
	INDEX_CONFIDENCE,   // 4
	INDEX_PROBILITY_CLASS_0
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
		b.w = 0.00001;

	data += stride;
	if (*data < 10.0)
		b.h = exp(*data) * anchor.height;
	else
		b.h = 0.00001;
	return b;
}
 
int YoloModule::EntryIndex(int batch, int anchor, int loc, int entry) {
	return batch * output.Elements3D() +
		((anchor * features + entry) * output.Elements2D()) + loc;
}
void YoloModule::DeltaBackground(int b, ObjectInfo* truths, int max_boxes, float& avg_anyobj) {
	float reciprocal_w = 1.0 / output.GetWidth();
	float reciprocal_h = 1.0 / output.GetHeight();
	for (int y = 0, offset = 0; y < output.GetHeight(); y++) {
		for (int x = 0; x < output.GetWidth(); x++, offset++) {
			for (int n = 0; n < masked_anchors.size(); n++) { // l.n == 3
															  // sigmoid(tx),sigmoid(ty),tw,th,Pr(Obj),Pr(Cls_0|Obj),Pr(Cls_0|Obj)...
				int box_index = EntryIndex(b, n, offset, INDEX_PROBILITY_X);
				Box pred = get_yolo_box(output_cpu + box_index, masked_anchors[n],
					x, y, reciprocal_w, reciprocal_h, output.Elements2D());
				float best_iou = 0;
				int best_t = 0;
				for (int t = 0; t < max_boxes; t++) {
					ObjectInfo& truth = truths[t];
					if (truth.class_id < 0) break;
					float iou = BoxIoU(pred, Box(truth));
					if (iou > best_iou) { best_iou = iou; best_t = t; }
				}
				// class_index : point to Pr(Obj)
				int object_index = box_index + (INDEX_CONFIDENCE - INDEX_PROBILITY_X) * output.Elements2D();
				avg_anyobj += output_cpu[object_index];

				//初始化全是背景，离目标较近的忽略，梯度为0
				if (best_iou > threshold_ignore)  delta_cpu[object_index] = 0;

				// 否则梯度是 -Pr(Obj)，这地方没有目标，将来输出的Pr(Obj)是0才对
				else delta_cpu[object_index] = 0.0f - output_cpu[object_index];

				// 更近的视为目标(不过这一步效果不太好，所以阈值设为1，永远不执行)
				if (best_iou > threshold_thruth) {
					delta_cpu[object_index] = 1 - output_cpu[object_index];
					ObjectInfo& truth = truths[best_t];//network->Truth(b, best_t);
					int class_id = truth.class_id;
					int class_index = object_index + output.Elements2D(); // * (INDEX_PROBILITY_CLASS_0 - INDEX_CONFIDENCE) ;
				 
					float temp = 0.0f;
					DeltaClass(class_id, class_index, temp);
					DeltaBox(truth, masked_anchors[n], box_index, x, y);

				}
			} // masked_anchor
		} // x
	}// y 
}
void YoloModule::DeltaBox(const ObjectInfo& truth, const AnchorBoxItem& anchor, int index, int x, int y) {

	float tx = (truth.x * output.GetWidth() - x);
	float ty = (truth.y * output.GetHeight() - y);
	float tw = log(truth.w / anchor.width);
	float th = log(truth.h / anchor.height);
	float scale = (2.0f - truth.w * truth.h);

	delta_cpu[index] = scale * (tx - output_cpu[index]);
	index += output.Elements2D();
	delta_cpu[index] = scale * (ty - output_cpu[index]);
	index += output.Elements2D();
	delta_cpu[index] = scale * (tw - output_cpu[index]);
	index += output.Elements2D();
	delta_cpu[index] = scale * (th - output_cpu[index]);
}
bool YoloModule::InitDescriptors(bool trainning) {
	if (output_cpu) {
		delete[]output_cpu;
		output_cpu = NULL;
	}
	if (delta_cpu) {
		delete[]delta_cpu;
		delta_cpu = NULL;
	}
	return true;
}
void YoloModule::DeltaClass(int class_id, int index, float & avg_cat) {

	int ti = index;
	if (class_id > 0)  ti += output.Elements2D() * class_id;
	if (delta_cpu[ti]) { //TOCHECK: always false?
		delta_cpu[ti] = 1 - output_cpu[ti];
		avg_cat += output_cpu[ti];
		return;
	}
	// Focal loss
	if (focal_loss) {
		// Focal Loss
		float alpha = 0.5f;    // 0.25 or 0.5
							   //float gamma = 2;    // hardcoded in many places of the grad-formula


		float grad = 1.0f, pt = output_cpu[ti];
		// http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d

		// http://blog.csdn.net/linmingan/article/details/77885832
		if (pt > 0.0f) grad = (pt - 1.0f) * (2.0f * pt * logf(pt) + pt - 1.0f);

		grad *= alpha;
		for (int n = 0; n < classes; n++, index += output.Elements2D()) {
			delta_cpu[index] = (((n == class_id) ? 1.0f : 0.0f) - output_cpu[index]);
			delta_cpu[index] *= grad;
			if (n == class_id) {
				avg_cat += output_cpu[index];
			}
		}
	}
	else {
		// default
		for (int n = 0; n < classes; n++, index += output.Elements2D()) {
			delta_cpu[index] = ((n == class_id) ? 1 : 0) - output_cpu[index];
			if (n == class_id && avg_cat) {
				avg_cat += output_cpu[index];
			}
		}
	}
}

bool YoloModule::Forward(ForwardContext& context) {
	if (!InferenceModule::Forward(context)) return false;
	int b, n;
	output = input;
	float* output_gpu = output.GetMem();
	bool ret = true;
	for (b = 0; b < output.GetBatch(); b++ ) {
		for (n = 0; n < masked_anchors.size(); n++) {
			int index = EntryIndex(b, n, 0, INDEX_PROBILITY_X);
			if (!activate_array_ongpu(output_gpu + index, 2 * output.Elements2D(), LOGISTIC)) {
				return false;
			}
			index = EntryIndex(b, n, 0, INDEX_CONFIDENCE);
			if (!activate_array_ongpu(output_gpu + index, (1 + classes) * output.Elements2D(), LOGISTIC))
				return false;
		}
	}
	if (!context.training) return true;

	if (NULL == delta_cpu) {
		delta_cpu = New float[output.MemElements()];
		cout << "alloc " << output.MemElements() << " float elements for delta_cpu.\n";
	}
	memset(delta_cpu, 0, output.MemBytes());
	float cost = 0.0;
	float avg_iou = 0, recall = 0, recall75 = 0, avg_cat = 0;
	float avg_obj = 0, avg_anyobj = 0;
	int count = 0, class_count = 0;
	float reciprocal_w = 1.0f / input.GetWidth(), reciprocal_h = 1.0f / output.GetHeight();
	if (NULL == output_cpu) {
		output_cpu = New float[output.MemElements()];
	}
	cudaError_t err = cudaMemcpy(output_cpu, output.GetMem(), output.MemBytes(), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cerr << "Error: cudaMemcpy ret " << err << " in YoloModule.Forward!\n" ;
		delete[]output_cpu;
		output_cpu = NULL;
		return false;
	}
	int  max_boxes = context.max_truths_per_batch;
	ObjectInfo* batch_truths = context.truths;
	for (int b = 0; b < output.GetBatch(); b++, batch_truths += max_boxes) {
		DeltaBackground(b, batch_truths, max_boxes, avg_anyobj);
		for (int t = 0; t < max_boxes; t++) {
			ObjectInfo& truth = batch_truths[t];
			if (truth.class_id < 0) break;
			float best_iou = 0;
			int n, best_n = 0, mask_n = -1;
			int x = (truth.x * output.GetWidth());
			int y = (truth.y * output.GetHeight());
			
			Box truth_shift(0, 0, truth.w, truth.h);
			float w, h;
			for (n = 0; n < GetNetwork().GetAnchorCount(); n++) {
				GetNetwork().GetAnchor(n, w, h);
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
				int offset = y * output.GetWidth() + x;
				int box_index = EntryIndex(b, mask_n, offset, INDEX_PROBILITY_X);

				DeltaBox(truth, masked_anchors[mask_n], box_index, x, y);

				int object_index = box_index + (INDEX_CONFIDENCE - INDEX_PROBILITY_X) * output.Elements2D();
				avg_obj += output_cpu[object_index];
				delta_cpu[object_index] = 1 - output_cpu[object_index];

				int class_id = truth.class_id;//context.truths[t*(4 + 1) + b*l.truths + 4];
 
				int class_index = object_index + output.Elements2D() /* *(INDEX_PROBILITY_CLASS_0 - INDEX_CONFIDENCE) */;
				DeltaClass(class_id, class_index, avg_cat);

				count++;
				class_count++;

				Box pred = get_yolo_box(output_cpu + box_index, masked_anchors[n],
					x, y, reciprocal_w, reciprocal_h, output.Elements2D());
				float iou = BoxIoU(pred, Box(truth));

				if (iou > 0.5f) recall += 1.0;
				if (iou > 0.75f) recall75 += 1.0;
				avg_iou += iou;
			}
		}
	}
	cost = square_sum_array(delta_cpu, output.MemElements());
	GetNetwork().RegisterLoss(cost);
	ostringstream oss;
	avg_anyobj /= output.MemElements();
	if (count > 0) {
		avg_iou /= count;
		avg_obj /= count;
		recall /= count;
		recall75 /= count;
		oss << "  Found " << count << " objects in detecting layer " << layer->GetIndex() << ".\n";
		if (class_count > 0) {
			avg_cat /= class_count;			
			oss << setprecision(4) << "   Avg IOU : " << avg_iou << ", Class : " << avg_cat << ", Obj : " << avg_obj;
			oss << ", No Obj : "<< avg_anyobj <<", 50% Recall : "<< recall <<", 75% Recall : "<< recall75 <<"\n" ;
		}
		else {
			oss << setprecision(4) << "   Avg IOU : " << avg_iou << ", Class : N.A. , Obj : " << avg_obj;
			oss << ", No Obj : " << avg_anyobj << ", 50% Recall : " << recall << ", 75% Recall : " << recall75 << "\n";
			 
		}
		cout << oss.str();
	}
	else
		cout << "  Found 0 objects in detecting layer "<<layer->GetIndex() <<".\n" ;

	return true;
}
bool YoloModule::Backward(FloatTensor4D& delta) {
	if (NULL == delta_cpu) return false;
	delta = output; 
	if (delta.MemElements() != output.MemElements()) return false;
	return cudaSuccess == cudaMemcpy(delta.GetMem(), delta_cpu, delta.MemBytes(), cudaMemcpyHostToDevice); 
}